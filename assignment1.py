import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from base import Base


class Assignment1(Base):

    def __init__(self):
        super(Assignment1, self).__init__()
        self.data = pickle.load(open('./features/cifar10/raw.pkl', 'rb'))
        print(self.data.shape)
        flat = self.data.reshape(len(self.data), -1)
        print(flat.shape)
        self.nn = NearestNeighbors(n_neighbors=1, metric=self.distance).fit(flat)
        # self.nn = NearestNeighbors(n_neighbors=1, metric='pyfunc', func=self.distance)

    def get_patch(self, tile):
        _, inds = self.nn.kneighbors(tile.reshape(1, -1))
        patch = self.data[inds[0]]
        return patch

    def feature(self, x):
        return np.mean(x.reshape(32, 32, 3), axis=(0, 1))

    def distance(self, x, y):
        return np.linalg.norm(self.feature(x) - self.feature(y))


if __name__ == '__main__':
    img = Image.open('./images/bolt.jpg').convert('RGB')
    target_image = np.array(img) / 255

    main = Assignment1()
    output_image = main.algorithm(target_image)

    plt.imshow(output_image)
    plt.show()
    plt.savefig('./mosaic.png')
