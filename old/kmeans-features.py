from sklearn.cluster import KMeans
import torch
import numpy as np
from PIL import Image
import os
import pickle

mu, _ = torch.load('./features2.pt')
num_images = len(mu)
num_features = mu.size(1)

print('Number of images: ', num_images)
print('Number of features per image: ', num_features)

model = KMeans(n_clusters=1000, random_state=0, n_jobs=10, verbose=True).fit(mu.numpy())
pickle.dump(model, open('./models-k1000-feat2.pkl', 'wb'))
