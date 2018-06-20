from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import torchvision
from torchvision import transforms
from model import VAE2
import numpy as np
import torch
from PIL import Image
import pickle
import random
import matplotlib.pyplot as plt


test_image = Image.open('/home/adrian/Documents/pytorch-mosaic/data/tiny-imagenet-200/val/images/val_2.JPEG').convert('RGB')
test_image = test_image.resize((32, 32))
array = np.array(test_image).reshape((1, -1))

autoencoder = torch.load('autoencoder_conv.pt').to('cpu')
model = pickle.load(open('./model-k1000-feat.pkl', 'rb'))
dataset = torchvision.datasets.CIFAR10('./data/cifar10/', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mu, _ = torch.load('features.pt')

print(model.labels_)
print(len(model.labels_))
# print(model.cluster_centers_)


closest, _ = pairwise_distances_argmin_min(model.cluster_centers_, mu)

# prediction = model.predict(array)

center = model.cluster_centers_[1] / 256


# plt.imshow(center.reshape((32, 32, 3)))
# plt.show()



def slice_image(image, tile_size):
    height = image.shape[0]
    width = image.shape[1]
    assert height > tile_size and width > tile_size

    num_tiles_y = height // tile_size
    num_tiles_x = width // tile_size

    height = num_tiles_y * tile_size
    width = num_tiles_x * tile_size

    image = image[:height, :width]

    print('new shape', image.shape)

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


def random_cluster_representative(cluster, data_labels):
    where = [i for i, x in enumerate(data_labels) if x == cluster]
    i = random.choice(where)
    return data_labels[i]


test_image = Image.open('./images/bolt.jpg').convert('RGB')
# test_image = test_image.resize((2 * test_image.width, 2 * test_image.height), Image.BILINEAR)
array = np.array(test_image)

print('old shape', array.shape)

tiles = slice_image(array, 32)
num_tiles = tiles.shape[0] * tiles.shape[1]

# plt.imshow(tiles[5, 5] / 256)
# plt.show()

x = torch.from_numpy(tiles).float().view(-1, 32, 32, 3).permute(0, 3, 1, 2)

with torch.no_grad():
    tile_features, _ = autoencoder.encode(x)

print('tile features: ', tile_features.size())

# flat_tiles = np.stack(tiles).reshape((num_tiles, -1))
prediction = model.predict(tile_features)

new_tiles = []
for p in prediction:
    # index = random_cluster_representative(p, model.labels_)
    # center = data[index]
    # center = model.cluster_centers_[p]
    center, _ = dataset[closest[p]]
    new_tiles.append(center.permute(1, 2, 0).numpy())

new_tiles = np.stack(new_tiles).reshape(tiles.shape)
new_image = make_image(new_tiles)

plt.subplot(1, 2, 1)
plt.imshow(new_image)
plt.subplot(1, 2, 2)
plt.imshow(array)
plt.show()
