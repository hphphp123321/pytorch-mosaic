from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
import os
import pickle


def load_train_data(folder):
    data = []
    for dirpath, _, filenames in os.walk(folder):
        image_names = [f for f in filenames if f.endswith(".JPEG")]
        for image_name in image_names:
            image = Image.open(os.path.join(dirpath, image_name))
            image = image.convert('RGB').resize((32, 32), Image.BILINEAR)
            array = np.array(image)
            data.append(array)

    return data


train_folder = "/home/adrian/Documents/pytorch-mosaic/data/tiny-imagenet-200/train/"
data = load_train_data(train_folder)
num_images = len(data)

data = np.stack(data).reshape((num_images, -1))
num_features = data.shape[1]

print('Number of images: ', num_images)
print('Number of features per image: ', num_features)

model = KMeans(n_clusters=1000, random_state=0, n_jobs=10, verbose=True).fit(data)

pickle.dump(model, open('./models.pkl', 'wb'))
pickle.dump(data, open('./data.pkl', 'wb'))
