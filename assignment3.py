import numpy as np
import pickle
import os
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import utils
import vae_train
import vae_encode
import torchvision
from torchvision import transforms

from base import Base
from vae import VAE


def main():
    """
    Assignment 3 - Neural Network Features

    """

    # The program will start execution here
    # Change the filename to load your favourite picture
    file = './images/1.jpg'
    train_features = False
    train = True

    img = Image.open(file).convert('RGB')
    img = utils.resize_proportional(img, new_height=900)
    target_image = np.array(img) / 255

    # This will execute the Mosaicking algorithm of Assignment 3
    main = Assignment3()
    main.encode_features(train_features)
    main.train(train)
    output_image = main.mosaic(target_image)

    # Saving the image inside in project root folder
    output_image *= 255
    im = Image.fromarray(output_image.astype('uint8'))
    im.save(utils.datetime_filename('output/A3/mosaics/mosaic.png'))


class Assignment3(Base):

    def __init__(self):
        super(Assignment3, self).__init__()
        self.data = self.get_data()
        self.kmeans_file = 'models/kmeans-feat.pkl'
        self.encoder_file = 'models/conv_vae.pt'
        self.feature_file = 'features/cifar10/vae.pt'
        self.kmeans = self.get_model()
        self.encoder = None
        self.features = None
        self.closest = None

    def get_data(self):
        transform = transforms.Compose([
            transforms.Resize([8, 8]),
            transforms.ToTensor()
        ])

        dataset = torchvision.datasets.CIFAR10(
            './data/cifar10/',
            train=True,
            transform=transform,
            target_transform=None,
            download=True
        )
        return dataset

    def get_model(self):
        """
        TO BE IMPLEMENTED BY STUDENT

        """
        kmeans = KMeans(
            n_clusters=500,
            n_init=1,
            max_iter=300,
            tol=0.0001,
            verbose=True,
            n_jobs=10,
        )

        return kmeans

    def train(self, train=True):
        if train:
            data = self.features
            self.kmeans.fit(data)
            self.closest, _ = pairwise_distances_argmin_min(self.kmeans.cluster_centers_, data)
            with open(self.kmeans_file, 'wb') as f:
                pickle.dump((self.kmeans, self.closest), f)
        else:
            with open(self.kmeans_file, 'rb') as f:
                self.kmeans, self.closest = pickle.load(f)

    def encode_features(self, train=True):
        if train:
            vae_train.train(self.encoder_file, self.data)

        self.encoder = torch.load(self.encoder_file).to('cpu')
        if not os.path.exists(self.feature_file) or train:
            vae_encode.encode(self.encoder_file, self.feature_file, self.data)

        self.features, _ = torch.load(self.feature_file)

    def get_patch(self, tile):
        x = torch.from_numpy(tile).float().view(1, 8, 8, 3).permute(0, 3, 1, 2)
        with torch.no_grad():
            self.encoder.eval()
            tile_features, _ = self.encoder.encode(x)

        cluster_indices = self.kmeans.predict(tile_features)
        patch, _ = self.data[self.closest[cluster_indices[0]]]
        patch = patch.permute(1, 2, 0).numpy()
        print(patch.shape)
        return patch


def make_folders():
    os.makedirs('output/A3/mosaics/', exist_ok=True)
    os.makedirs('models', exist_ok=True)


if __name__ == '__main__':
    make_folders()
    main()
