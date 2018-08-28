import torch
import os
import torch
import torchvision
from torchvision import transforms
from model import VAE2



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = torchvision.datasets.CIFAR100('./data/cifar100/', train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=1,
                                          shuffle=False)


model = torch.load('autoencoder_conv2.pt')

with torch.no_grad():
    model.eval()
    data_mu = []
    data_logvar = []
    for i, (x, _) in enumerate(data_loader):
        x = x.to(device)
        mu, logvar = model.encode(x)

        data_mu.append(mu)
        data_logvar.append(logvar)

    mu = torch.cat(data_mu).cpu()
    logvar = torch.cat(data_logvar).cpu()


torch.save((mu, logvar), 'features2.pt')
