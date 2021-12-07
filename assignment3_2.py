import numpy as np
import pickle
import os
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors

import utils
import dae_train
import dae_encode
import torchvision
from torchvision import transforms

from base import Base


def main():
    """
    Assignment 3 - Neural Network Features

    """

    # The program will start execution here
    # Change the filename to load your favourite picture
    file = 'images/1.jpg'
    train_features = False
    train = False

    img = Image.open(file).convert('RGB')
    img = utils.resize_proportional(img, new_height=1000)
    target_image = np.array(img) / 255

    # This will execute the Mosaicking algorithm of Assignment 3
    main = Assignment3()
    main.encode_features(train_features)
    main.train(train)
    output_image = main.mosaic(target_image)

    # Saving the image inside in project root folder
    output_image *= 255
    im = Image.fromarray(output_image.astype('uint8'))
    im.save(utils.datetime_filename('output/A3_2/mosaics/mosaic.png'))


class Assignment3(Base):

    def __init__(self):
        super(Assignment3, self).__init__()
        self.data = self.get_data()
        self.kmeans_file = 'models/kmeans-feat-dae.pkl'
        self.encoder_file = 'models/conv_dae.pt'
        self.feature_file = 'features/cifar10/dae.pt'
        self.kmeans = self.get_model()
        self.encoder = None
        self.features = None
        self.closest = None

    def get_data(self):
        dataset = torchvision.datasets.CIFAR10(
            './data/cifar10/',
            train=True,
            transform=transforms.ToTensor(),
            target_transform=None,
            download=True
        )
        return dataset

    def get_model(self):
        """
        TO BE IMPLEMENTED BY STUDENT

        """
        # kmeans = KMeans(
        #     n_clusters=500,
        #     n_init=1,
        #     max_iter=50,
        #     tol=0.01,
        #     verbose=True,
        #     n_jobs=10,
        # )
        kmeans = NearestNeighbors(n_neighbors=1)
        return kmeans

    def train(self, train=True):
        if train:
            print('Training Kmeans')
            data = self.features
            self.kmeans.fit(data)
            # self.closest, _ = pairwise_distances_argmin_min(self.kmeans.cluster_centers_, data)
            self.closest = []
            with open(self.kmeans_file, 'wb') as f:
                pickle.dump((self.kmeans, self.closest), f)
            print('done.')
        else:
            with open(self.kmeans_file, 'rb') as f:
                self.kmeans, self.closest = pickle.load(f)

    def encode_features(self, train=True):
        if train:
            dae_train.train(self.encoder_file, self.data)

        encoder, _ = torch.load(self.encoder_file)
        self.encoder = encoder.to('cpu')

        if not os.path.exists(self.feature_file) or train:
            print('Encoding features ...')
            dae_encode.encode(self.encoder_file, self.feature_file, self.data)
            print('done.')

        print('Loading features ...')
        self.features = torch.load(self.feature_file)
        print('done.')

    def get_patch(self, tile):
        x = torch.from_numpy(tile).float().view(1, 32, 32, 3).permute(0, 3, 1, 2)
        with torch.no_grad():
            self.encoder.eval()
            tile_features = self.encoder(x)

        _, inds = self.kmeans.kneighbors(tile_features.reshape(1, -1))
        patch, _ = self.data[inds[0][0]]
        # cluster_indices = self.kmeans.predict(tile_features)
        # # patch, _ = self.data[self.closest[cluster_indices[0]]]
        # patch, _ = self.data[cluster_indices[0]]
        patch = patch.permute(1, 2, 0).numpy()
        return patch


def make_folders():
    os.makedirs('output/A3_2/mosaics/', exist_ok=True)
    os.makedirs('models', exist_ok=True)


if __name__ == '__main__':
    make_folders()
    main()
