# Simple Convolutional Autoencoder
# Code by GunhoChoi
import os

import torch
import torch.nn as nn
import torch.utils as utils
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Set Hyperparameters
from dae import Encoder, Decoder


def train(save_file, dataset, epochs=10):
    sample_dir = './output/A3_test/dae_train/'
    os.makedirs(sample_dir, exist_ok=True)

    batch_size = 100
    learning_rate = 0.0002

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    encoder = Encoder(batch_size).cuda()
    decoder = Decoder(batch_size).cuda()

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

            """
            TO BE IMPLEMENTED BY STUDENT
            
            Feed the input image to the encoder. 
            Take output of the encoder and feed it to the decoder.
            
            Finally compute the loss (error) between the output and the input.
            
            """
            output = None
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
