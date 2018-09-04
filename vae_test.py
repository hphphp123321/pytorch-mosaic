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
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Hyper-parameters
image_size = 32 * 32 * 3
z_dim = 512
num_epochs = 100
batch_size = 128
learning_rate = 1e-3


dataset = torchvision.datasets.CIFAR100('./data/cifar10/', train=False, transform=transforms.ToTensor(), target_transform=None, download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False
)


model = torch.load('models/conv_vae.pt').to(device)

with torch.no_grad():
    for i, (x, _) in enumerate(data_loader):
        x = x.to(device)
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 3, 32, 32), out.view(-1, 3, 32, 32)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'test-batch-{}.png'.format(i)))

