import os

import numpy as np
import pickle
from sklearn.externals import joblib
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import utils
from base import Base


def main():
    """
    Assignment 1a - Average Patch Features
    Assignment 1b - Nearest Neighbor Search

    """

    # The program will start execution here
    # Change the filename to load your favourite picture
    file = './images/eye2.jpg'

    # Setting this to True will train the model (or pre-compute the features)
    # All models are automatically saved in the folder 'models'
    # After the model is trained well, you can set this to false
    train_features = False
    train_model = False


    # Load image and resize it to a fixed size (keeping aspect ratio)
    img = Image.open(file).convert('RGB')
    img = utils.resize_proportional(img, new_height=900)
    target_image = np.array(img) / 255

    # This will execute the Mosaicking algorithm of Assignment 1
    main = Assignment1()
    main.encode_features(train_features)
    main.train(train_model)
    output_image = main.mosaic(target_image)

    # Saving the image inside in project root folder
    output_image *= 255
    im = Image.fromarray(output_image.astype('uint8'))
    im.save(utils.datetime_filename('output/A1/mosaics/mosaic.png'))


class Assignment1(Base):

    def __init__(self):
        super(Assignment1, self).__init__()
        self.data = pickle.load(open('./features/cifar10/raw.pkl', 'rb'))
        self.features = None
        self.nn = self.get_model()
        self.model_file = './models/nearest_neighbor.pkl'
        self.feature_file = './features/cifar10/avg.pkl'

    def get_model(self):
        """
        TO BE IMPLEMENTED BY STUDENT

        """
        return NearestNeighbors(n_neighbors=1, metric=self.distance)

    def get_patch(self, tile):
        tile_feature = self.feature(tile)
        _, inds = self.nn.kneighbors(tile_feature.reshape(1, -1))
        patch = self.data[inds[0]]
        return patch

    def get_patches(self, tiles):
        # N x (32 * 32 * 3)
        tile_features = self.feature(tiles)  # N x feat_size
        _, inds = self.nn.kneighbors(tile_features)
        patches = self.data[inds]
        return patches

    def encode_features(self, train=True):
        if train:
            self.features = np.stack([self.feature(patch) for patch in self.data])
            with open(self.feature_file, 'wb') as file:
                pickle.dump(self.features, file)
        else:
            with open(self.feature_file, 'rb') as file:
                self.features = pickle.load(file)

    def feature(self, x):
        """
        TO BE IMPLEMENTED BY STUDENT

        Compute the average color across the patch x.

        :param x: The image patch of size 32 x 32 x 3 flattened as a long vector of size 1 x 3072
        :return: The average pixel color
        """

        return np.mean(x.reshape(32, 32, 3), axis=(0, 1))

    def distance(self, x, y):
        """
        TO BE IMPLEMENTED BY STUDENT

        Compute the distance between features x and y.
        The distance is used for nearest neighbor search in feature space (avg. color feature)

        """
        return np.linalg.norm(x - y)

    def train(self, train=True):
        if train:
            print('Fitting NN model ...')
            self.nn.fit(self.features.reshape(len(self.features), -1))
            # with open(self.model_file, 'wb') as f:
            joblib.dump(self.nn, self.model_file)

            print('... done.')
        elif os.path.exists(self.model_file):
            # with open(self.model_file, 'rb') as f:
            self.nn = joblib.load(self.model_file)
        else:
            print('Model not found.')


def make_folders():
    os.makedirs('output/A1/mosaics/', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('features/cifar10', exist_ok=True)


if __name__ == '__main__':
    make_folders()
    main()

