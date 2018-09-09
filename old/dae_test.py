# Simple Convolutional Autoencoder
# Code by GunhoChoi

import torch
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

mnist_test = dset.CIFAR10("data/cifar10", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
encoder, decoder = torch.load('./models/deno_autoencoder.pkl')
encoder.batch_size = 1
decoder.batch_size = 1

for image, _ in mnist_test:

    image = Variable(image.unsqueeze(0)).cuda()

    output = encoder(image)
    output = decoder(output)

    img = image[0].cpu()
    input_img = image[0].cpu()
    output_img = output[0].cpu()

    origin = img.data.numpy()
    inp = input_img.data.numpy()
    out = output_img.data.numpy()

    plt.imshow(origin[0].tr)
    plt.show()

    plt.imshow(inp[0] * 255)
    plt.show()

    plt.imshow(out[0] * 255)
    plt.show()
