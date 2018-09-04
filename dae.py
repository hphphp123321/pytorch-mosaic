# Simple Convolutional Autoencoder
# Code by GunhoChoi

import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Set Hyperparameters
from dae_model import Encoder, Decoder

epoch = 100
batch_size = 100
learning_rate = 0.0002

# Download Data

# mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
# mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

mnist_train = dset.CIFAR10("data/cifar10", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dset.CIFAR10("data/cifar10", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

# Set Data Loader(input pipeline)

train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)


# Encoder
# torch.nn.Conv2d(in_channels, out_channels, kernel_size,
#                 stride=1, padding=0, dilation=1,
#                 groups=1, bias=True)
# batch x 1 x 28 x 28 -> batch x 512

# Encoder
# torch.nn.Conv2d(in_channels, out_channels, kernel_size,
#                 stride=1, padding=0, dilation=1,
#                 groups=1, bias=True)
# batch x 1 x 28 x 28 -> batch x 512


encoder = Encoder(batch_size).cuda()


# Decoder
# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
#                          stride=1, padding=0, output_padding=0,
#                          groups=1, bias=True)
# output_height = (height-1)*stride + kernel_size - 2*padding + output_padding
# batch x 512 -> batch x 1 x 28 x 28

# Decoder
# torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
#                          stride=1, padding=0, output_padding=0,
#                          groups=1, bias=True)
# output_height = (height-1)*stride + kernel_size - 2*padding + output_padding
# batch x 512 -> batch x 1 x 28 x 28


decoder = Decoder(batch_size).cuda()

# Noise

noise = torch.rand(batch_size, 1, 32, 32)

# loss func and optimizer
# we compute reconstruction after decoder so use Mean Squared Error
# In order to use multi parameters with one optimizer,
# concat parameters after changing into list

parameters = list(encoder.parameters()) + list(decoder.parameters())
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(parameters, lr=learning_rate)

# train encoder and decoder

for i in range(epoch):
    for image, label in train_loader:
        # image_n = torch.mul(image + 0.25, 0.1 * noise)
        image = Variable(image).cuda()
        # image_n = Variable(image_n).cuda()
        optimizer.zero_grad()
        output = encoder(image)
        output = decoder(output)
        loss = loss_func(output, image)
        loss.backward()
        optimizer.step()
        # break
        print(loss.item())

    torch.save([encoder, decoder], './models/deno_autoencoder.pkl')

    # check image with noise and denoised image\

    img = image[0].cpu()
    input_img = image[0].cpu()
    output_img = output[0].cpu()

    origin = img.data.numpy()
    inp = input_img.data.numpy()
    out = output_img.data.numpy()

    plt.imshow(origin[0])
    plt.show()

    plt.imshow(inp[0])
    plt.show()

    plt.imshow(out[0])
    plt.show()

    print(label[0])