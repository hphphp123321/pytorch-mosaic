import utils
import numpy as np
import matplotlib.pyplot as plt
import os
import utils

from assignment1 import Assignment1


def make_folders():
    os.makedirs('output/A1_test/', exist_ok=True)


def test_grid():
    patch_size = 32
    original_width = 100
    original_height = 120

    nx, ny = utils.number_of_patches(original_width, original_height, patch_size)
    new_width, new_height = utils.output_image_size(nx, ny, patch_size)

    print('new width is ' + str(new_width))
    print('new height is ' + str(new_height))


def test_feature():
    """
    TO BE COMPLETED BY STUDENT

    Change the code so that multiple columns are displayed.
    Each column should contain a different (randomly picked) patch and it's mean color.
    Run the script first to see what the output is.

    """
    num_cols = 3  # CHANGE THIS

    i = 0
    for col in range(num_cols):
        i += 1
        # patch_idx = np.random.randint(0, len(main.data))
        patch_idx = 11020
        patch = main.data[patch_idx]
        feature = main.feature(patch)
        patch_mean = feature.reshape(1, 1, 3).repeat(32, 0).repeat(32, 1)

        # grid plot of size 2 x num_cols
        plt.subplot(2, num_cols, i)
        plt.title(str(patch_idx))
        plt.imshow(patch)

        plt.subplot(2, num_cols, num_cols + i)
        plt.imshow(patch_mean)

    fig = plt.gcf()
    plt.show()
    fname = utils.datetime_filename('output/A1_test/features.png')
    fig.savefig(fname, format='png', dpi=300)


def test_distance():
    patch1 = main.data[0]
    patch2 = main.data[1]
    dist = main.distance(patch1, patch1)
    print('Distance between same patches: {:.4f}'.format(dist))

    dist = main.distance(patch1, patch2)
    print('Distance between two different patches: {:.4f}'.format(dist))


if __name__ == '__main__':
    main = Assignment1()
    main.train(False)
    make_folders()

    print('Test Grid')
    test_grid()

    print('\nTest Feature')
    test_feature()

    print('\nTest Distance')
    test_distance()
