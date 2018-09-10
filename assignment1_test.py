import numpy as np
import matplotlib.pyplot as plt
import os
import utils
from assignment1 import Assignment1


def make_folders():
    os.makedirs('output/A1_test/features', exist_ok=True)
    os.makedirs('output/A1_test/neighbors', exist_ok=True)


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
        patch_idx = 11020  # CHANGE THIS
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
    fname = utils.datetime_filename('output/A1_test/features/grid.png')
    fig.savefig(fname, format='png', dpi=300)


def test_distance():
    patch1 = main.data[0]
    patch2 = main.data[1]
    dist = main.distance(main.feature(patch1), main.feature(patch1))
    print('Distance between same patches: {:.4f}'.format(dist))

    dist = main.distance(main.feature(patch1), main.feature(patch2))
    print('Distance between two different patches: {:.4f}'.format(dist))


def test_neighbors():
    num_cols = 3
    num_rows = 3

    main.nn.n_neighbors = num_cols
    i = 0

    for row in range(num_rows):
        patch_idx = np.random.randint(0, len(main.data))
        patch = main.data[patch_idx]
        neighbors = main.get_patch(patch).reshape(-1, 32, 32, 3)

        for col in range(num_cols):
            i += 1

            if col == 0:
                plt.subplot(num_rows, num_cols, i)
                plt.title(str(patch_idx))
                plt.imshow(patch)
            else:
                plt.subplot(num_rows, num_cols, i)
                # plt.title(str(patch_idx))
                plt.imshow(neighbors[col])

    fig = plt.gcf()
    plt.show()
    fname = utils.datetime_filename('output/A1_test/neighbors/grid.png')
    fig.savefig(fname, format='png', dpi=300)


if __name__ == '__main__':
    main = Assignment1()
    main.encode_features(False)
    main.train(False)
    make_folders()

    print('Test Grid')
    test_grid()

    print('\nTest Feature')
    test_feature()

    print('\nTest Distance')
    test_distance()

    print('\nTest Neighbors')
    test_neighbors()
