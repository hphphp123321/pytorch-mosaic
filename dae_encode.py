import torch
import os
import torch
import torchvision
from torchvision import transforms


def encode(model_file, features_file, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False
    )
    encoder, decoder = torch.load(model_file)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        features = []
        for i, (x, _) in enumerate(data_loader):
            x = x.to(device)
            out = encoder(x)
            features.append(out.cpu())
            print(i, len(data_loader))

        features = torch.cat(features)

    torch.save(features, features_file)


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(
        './data/cifar10/',
        train=True,
        transform=transforms.ToTensor(),
        target_transform=None,
        download=True
    )
    model_file = 'models/conv_dae.pt'
    features_file = 'features/cifar10/dae.pt'
    encode(model_file, features_file, dataset)