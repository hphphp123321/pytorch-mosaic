import numpy as np

from utils import slice_image, make_image


class Base:

    def __init__(self):
        pass

    def mosaic(self, image):
        tiles = slice_image(image, 32)
        num_tiles = tiles.shape[0] * tiles.shape[1]
        flat_tiles = np.stack(tiles).reshape(num_tiles, -1)
        new_tiles = []
        # print(flat_tiles[0])
        for i, tile in enumerate(flat_tiles):
            print(f'tile {i:d} / {num_tiles:d}')
            patch = self.get_patch(tile)
            new_tiles.append(patch.reshape(32, 32, 3))

        # print(new_tiles[0])
        new_tiles = np.stack(new_tiles).reshape(tiles.shape)
        new_image = make_image(new_tiles)
        return new_image

    def train(self):
        pass

    def get_patch(self, tile):
        pass

    def feature(self, x):
        pass

    def distance(self, x, y):
        pass
