# Simple Convolutional Autoencoder
# Code by GunhoChoi
import os

import torch
import torch.nn as nn
import torch.utils as utils
import torchvision
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# Set Hyperparameters
from dae import Encoder, Decoder


def train(save_file, dataset, epochs=10):
    sample_dir = './output/A3_test/dae_train/'
    os.makedirs(sample_dir, exist_ok=True)

    batch_size = 100
    learning_rate = 0.0002

    # Download Data

    # mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    # mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

    # mnist_train = dset.CIFAR10("./data/cifar10", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    # mnist_test = dset.CIFAR10("./data/cifar10", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

    # Set Data Loader(input pipeline)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


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

    for epoch in range(epochs):
        for i, (image, label)in enumerate(train_loader):
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
            if (i+1) % 10 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}".format(
                    epoch + 1, epochs, i + 1, len(train_loader), loss.item()))

        torch.save([encoder, decoder], save_file)

        with torch.no_grad():
            # Save the reconstructed images
            output = encoder(image)
            output = decoder(output)
            x_concat = torch.cat([image.view(-1, 3, 32, 32), output.view(-1, 3, 32, 32)], dim=3)
            save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))


        # img = image[0].cpu()
        # input_img = image[0].cpu()
        # output_img = output[0].cpu()
        #
        # origin = img.data.numpy()
        # inp = input_img.data.numpy()
        # out = output_img.data.numpy()
        #
        # plt.imshow(origin[0] * 255)
        # plt.show()
        #
        # plt.imshow(inp[0] * 255)
        # plt.show()
        #
        # plt.imshow(out[0] * 255)
        # plt.show()
        #
        # print(label[0])


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(
        './data/cifar10/',
        train=True,
        transform=transforms.ToTensor(),
        target_transform=None,
        download=True
    )

    model_file = './models/conv_dae.pt'
    train(model_file, dataset)
