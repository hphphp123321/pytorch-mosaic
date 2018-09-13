# # Simple Convolutional Autoencoder
# # Code by GunhoChoi
#
# import torch
# from torch.autograd import Variable
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
#
# mnist_test = dset.CIFAR10("data/cifar10", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
# encoder, decoder = torch.load('./models/deno_autoencoder.pkl')
# encoder.batch_size = 1
# decoder.batch_size = 1
#
# for image, _ in mnist_test:
#
#     image = Variable(image.unsqueeze(0)).cuda()
#
#     output = encoder(image)
#     output = decoder(output)
#
#     img = image[0].cpu()
#     input_img = image[0].cpu()
#     output_img = output[0].cpu()
#
#     origin = img.data.numpy()
#     inp = input_img.data.numpy()
#     out = output_img.data.numpy()
#
#     plt.imshow(origin[0].tr)
#     plt.show()
#
#     plt.imshow(inp[0] * 255)
#     plt.show()
#
#     plt.imshow(out[0] * 255)
#     plt.show()

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from vae import VAE

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
sample_dir = './output/A3_test/dae_test/'
os.makedirs(sample_dir, exist_ok=True)

# Hyper-parameters
image_size = 32 * 32 * 3
z_dim = 512
num_epochs = 100
batch_size = 100
learning_rate = 1e-3

dataset = torchvision.datasets.CIFAR10('./data/cifar10/', train=False, transform=transforms.ToTensor(),
                                        target_transform=None, download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
)

encoder, decoder = torch.load('models/conv_dae.pt')
encoder = encoder.to(device)
decoder = decoder.to(device)

with torch.no_grad():
    encoder.eval()
    decoder.eval()
    for i, (x, _) in enumerate(data_loader):
        x = x.to(device)
        out = encoder(x)
        out = decoder(out)
        x_concat = torch.cat([x.view(-1, 3, 32, 32), out.view(-1, 3, 32, 32)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'test-batch-{}.png'.format(i)))
