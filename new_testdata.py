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


def select_and_rename_images(source_dir, target_dir1, target_dir2, label, count=500):
    if not os.path.exists(target_dir1):
        os.makedirs(target_dir1)
    if not os.path.exists(target_dir2):
        os.makedirs(target_dir2)
    images = [img for img in os.listdir(source_dir) if img.endswith('.jpg')]
    selected_images = random.sample(images, min(count, len(images)))
    for i, img in enumerate(selected_images):
        new_img_name = f'{label}_{img}'
        img_path = os.path.join(source_dir, img)
        image = Image.open(img_path)
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        image.save(os.path.join(target_dir1, new_img_name))
        noisy_image = add_noise(image)
        noisy_image.save(os.path.join(target_dir2, new_img_name))


if __name__ == '__main__':
    select_and_rename_images('dataset/violence_dataset/non_violence',
                             'dataset/test_new', 'dataset/test_new_noise', 0)
    select_and_rename_images('dataset/violence_dataset/violence',
                             'dataset/test_new', 'dataset/test_new_noise', 1)
