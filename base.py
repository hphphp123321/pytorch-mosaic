import numpy as np

from utils import slice_image, make_image


class Base:

    def __init__(self):
        pass

    def mosaic(self, image):
        tiles = slice_image(image, 8)
        num_tiles = tiles.shape[0] * tiles.shape[1]
        flat_tiles = np.stack(tiles).reshape(num_tiles, -1)
        new_tiles = []
        # print(flat_tiles[0])
        for i, tile in enumerate(flat_tiles):
            print(f'tile {i:d} / {num_tiles:d}')
            patch = self.get_patch(tile)
            new_tiles.append(patch.reshape(8, 8, 3))

        # print(new_tiles[0])
        new_tiles = np.stack(new_tiles).reshape(tiles.shape)
        new_image = make_image(new_tiles)
        return new_image

    def mosaic_fast(self, image):
        print('Running "fast" mosaic ...')
        tiles = slice_image(image, 32)
        num_tiles = tiles.shape[0] * tiles.shape[1]
        flat_tiles = np.stack(tiles).reshape(num_tiles, -1)
        patches = self.get_patches(flat_tiles)
        new_tiles = np.stack(patches).reshape(tiles.shape)
        new_image = make_image(new_tiles)
        return new_image

    def train(self):
        pass

    def encode_features(self, train=True):
        pass

    def get_patch(self, tile):
        pass

    def get_patches(self, tiles):
        pass

    def feature(self, x):
        pass

    def distance(self, x, y):
        pass
