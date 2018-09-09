import random

import utils
import numpy as np
import matplotlib.pyplot as plt
import os
import utils

from assignment2 import Assignment2


def make_folders():
    os.makedirs('output/A2_test/clusters', exist_ok=True)


# def test_feature():
#     """
#     TO BE COMPLETED BY STUDENT
#
#     Change the code so that multiple columns are displayed.
#     Each column should contain a different (randomly picked) patch and it's mean color.
#     Run the script first to see what the output is.
#
#     """
#     num_cols = 3  # CHANGE THIS
#
#     i = 0
#     for col in range(num_cols):
#         i += 1
#         # patch_idx = np.random.randint(0, len(main.data))
#         patch_idx = 11020  # CHANGE THIS
#         patch = main.data[patch_idx]
#         feature = main.feature(patch)
#         patch_mean = feature.reshape(1, 1, 3).repeat(32, 0).repeat(32, 1)
#
#         # grid plot of size 2 x num_cols
#         plt.subplot(2, num_cols, i)
#         plt.title(str(patch_idx))
#         plt.imshow(patch)
#
#         plt.subplot(2, num_cols, num_cols + i)
#         plt.imshow(patch_mean)
#
#     fig = plt.gcf()
#     plt.show()
#     fname = utils.datetime_filename('output/A1_test/features/grid.png')
#     fig.savefig(fname, format='png', dpi=300)


def test_cluster():
    clusters = [5, 6, 7]
    num_cols = 5
    num_rows = len(clusters)

    i = 0
    for row in range(num_rows):
        cluster = clusters[row]
        cluster_center = main.data[main.closest[cluster]]
        labels = main.kmeans.labels_
        cluster_indices = [i for i, l in enumerate(labels) if l == cluster]

        d = main.kmeans.transform(main.data.reshape(len(main.data), -1))[:, cluster]
        ind = np.argsort(d)[:num_cols]
        closest = main.data[ind]

        for col in range(num_cols):
            i += 1

            # else:
            # patch_idx = random.choice(cluster_indices)
            # patch = main.data[patch_idx]
            patch = closest[col]
            # grid plot of size 2 x num_cols
            plt.subplot(num_rows, num_cols, i)
            if col == 0:
                plt.title('Center')
            # plt.title(str(patch_idx))
            plt.imshow(patch)

    fig = plt.gcf()
    plt.show()
    fname = utils.datetime_filename('output/A2_test/clusters/samples.png')
    fig.savefig(fname, format='png', dpi=300)


if __name__ == '__main__':
    main = Assignment2()
    main.train(False)
    make_folders()

    print('\nTest Clusters')
    test_cluster()
