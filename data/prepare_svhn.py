import torchvision
import torch
from torchvision.transforms import transforms
import pickle
import os


dataset = torchvision.datasets.SVHN(
    './svhn/',
    split='train',
    transform=transforms.ToTensor(),
    target_transform=None,
    download=True
)

data = [x for x, _ in dataset]
data = torch.stack(data).permute(0, 2, 3, 1).numpy()
# shape of data is (?, 32, 32, 3)

os.makedirs('../features/svhn', exist_ok=True)
with open('../features/svhn/raw.pkl', 'wb') as file:
    pickle.dump(data, file)
