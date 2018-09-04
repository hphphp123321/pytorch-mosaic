import torch
import os
import torch
import torchvision
from torchvision import transforms
from vae import VAE


def encode(model_file, features_file, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False
    )
    model = torch.load(model_file)

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

    torch.save((mu, logvar), features_file)


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(
        './data/cifar10/',
        train=True,
        transform=transforms.ToTensor(),
        target_transform=None,
        download=True
    )
    model_file = 'models/conv_vae.pt'
    features_file = 'features/cifar10/vae.pt'
    encode(model_file, features_file, dataset)
