import utils


def test_grid():
    patch_size = 32
    original_width = 100
    original_height = 120

    nx, ny = utils.number_of_patches(original_width, original_height, patch_size)
    new_width, new_height = utils.output_image_size(nx, ny, patch_size)

    print('new width is ' + str(new_width))
    print('new height is ' + str(new_height))


def test_feature():
    pass


def test_distance():
    pass


if __name__ == '__main__':
    test_grid()