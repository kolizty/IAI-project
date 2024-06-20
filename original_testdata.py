import os
import random
from PIL import Image
import numpy as np


def add_noise(image):
    image_np = np.array(image)
    noise = np.random.randint(-25, 25, image_np.shape, dtype='int16')
    noisy_image_np = image_np + noise
    noisy_image_np = np.clip(noisy_image_np, 0, 255)
    noisy_image = Image.fromarray(noisy_image_np.astype('uint8'))
    return noisy_image


def create_test_datasets(train_dir, test_dir, test_noise_dir, test_size=0.1):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(test_noise_dir):
        os.makedirs(test_noise_dir)
    images = [f for f in os.listdir(train_dir) if f.endswith('.jpg')]
    n_test = int(len(images) * test_size)
    test_images = random.sample(images, n_test)

    for image_name in test_images:
        img_path = os.path.join(train_dir, image_name)
        image = Image.open(img_path)
        image.save(os.path.join(test_dir, image_name))
        noisy_image = add_noise(image)
        noisy_image.save(os.path.join(test_noise_dir, image_name))
        os.remove(img_path)


if __name__ == '__main__':
    create_test_datasets('dataset/train', 'dataset/test_original',
                         'dataset/test_noise', test_size=0.1)
