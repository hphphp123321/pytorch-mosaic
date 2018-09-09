import os

import numpy as np
import pickle
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import utils
from base import Base


def main():
    """
    Assignment 2 - Clustering

    """

    # The program will start execution here
    # Change the filename to load your favourite picture
    file = './images/dog1.png'
    train = False

    img = Image.open(file).convert('RGB')
    target_image = np.array(img) / 255

    # This will execute the Mosaicking algorithm of Assignment 2
    main = Assignment2()
    main.train(train)
    output_image = main.mosaic(target_image)

    # Saving the image inside in project root folder
    output_image *= 255
    im = Image.fromarray(output_image.astype('uint8'))
    im.save(utils.datetime_filename('output/A2/mosaics/mosaic.png'))


class Assignment2(Base):

    def __init__(self):
        super(Assignment2, self).__init__()
        self.data = pickle.load(open('./features/cifar10/raw.pkl', 'rb'))
        self.kmeans = self.get_model()
        self.closest = None

    def get_model(self):
        """
        TO BE IMPLEMENTED BY STUDENT

        """
        kmeans = KMeans(
            n_clusters=700,
            n_init=10,
            max_iter=500,
            tol=0.0001,
            verbose=True,
            n_jobs=10,
        )

        return kmeans

    def train(self, train=True):
        model_file = 'models/kmeans.pkl'
        if train:
            data = self.data.reshape(len(self.data), -1)
            self.kmeans.fit(data)
            self.closest, _ = pairwise_distances_argmin_min(self.kmeans.cluster_centers_, data)
            with open(model_file, 'wb') as f:
                pickle.dump((self.kmeans, self.closest), f)
        else:
            with open(model_file, 'rb') as f:
                self.kmeans, self.closest = pickle.load(f)

    def get_patch(self, tile):
        cluster_indices = self.kmeans.predict(tile.reshape(1, -1))
        patch = self.data[self.closest[cluster_indices[0]]]
        return patch


def make_folders():
    os.makedirs('output/A2/mosaics/', exist_ok=True)


if __name__ == '__main__':
    make_folders()
    main()

