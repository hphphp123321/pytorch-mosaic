import random

import utils
import numpy as np
import matplotlib.pyplot as plt
import os
import utils

from assignment2 import Assignment2


def make_folders():
    os.makedirs('output/A2_test/clusters', exist_ok=True)


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
