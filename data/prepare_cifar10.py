import torchvision
import torch
from torchvision.transforms import transforms
import pickle
import os


dataset = torchvision.datasets.CIFAR10(
    './cifar10/',
    train=True,
    transform=transforms.ToTensor(),
    target_transform=None,
    download=True
)

data = [x for x, _ in dataset]
data = torch.stack(data).permute(0, 2, 3, 1).numpy()
# shape of data is (50000, 32, 32, 3)

os.makedirs('../features/cifar10', exist_ok=True)
with open('../features/cifar10/raw.pkl', 'wb') as file:
    pickle.dump(data, file)
