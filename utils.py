import os
from datetime import datetime
from PIL import Image

import numpy as np


def number_of_patches(width, height, patch_size):
    """
    TO BE IMPLEMENTED BY STUDENT

    """
    return 0, 0


def output_image_size(n_patches_x, n_patches_y, patch_size):
    """
    TO BE IMPLEMENTED BY STUDENT

    """
    return 0, 0


def slice_image(image, tile_size):
    """
    TO BE COMPLETED BY STUDENT

    """
    height = image.shape[0]
    width = image.shape[1]
    assert height > tile_size and width > tile_size

    num_tiles_x, num_tiles_y = 0, 0  # CHANGE THIS
    width, height = 0, 0  # CHANGE THIS

    # Crop image to new size
    image = image[:height, :width]

    tiles = np.zeros((num_tiles_y, num_tiles_x, tile_size, tile_size, 3))
    for i, ty in enumerate(range(0, height, tile_size)):
        for j, tx in enumerate(range(0, width, tile_size)):
            tiles[i, j] = image[ty : ty + tile_size, tx : tx + tile_size]

    return tiles


def make_image(tiles):
    tile_size = tiles.shape[2]
    height = tiles.shape[0] * tile_size
    width = tiles.shape[1] * tile_size
    image = np.zeros((height, width, 3))

    for i, ty in enumerate(range(0, height, tile_size)):
        for j, tx in enumerate(range(0, width, tile_size)):
            image[ty : ty + tile_size, tx : tx + tile_size] = tiles[i, j]

    return image


def datetime_filename(filename):
    base, ext = os.path.splitext(filename)
    dtime = datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")
    return base + dtime + ext


def resize_proportional(pil_image, new_height):
    width, height = pil_image.size
    aspect = width / height
    img = pil_image.resize((int(new_height * aspect), int(new_height)), Image.BILINEAR)
    return img
