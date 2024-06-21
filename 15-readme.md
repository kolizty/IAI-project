## Introduction

This `classify.py` defines the `ViolenceClass` for violence image classification. It uses `model_path` to initialize the `ViolenceClass` by loading from saved checkpoint model. It  uses `n*3*224*224` pytorch tensor as input and outputs a length `n` python list as prediction, `0` for non-violence and `1` for violence.



The `classify.py` only requires `model.py` for model information and `violence_predict.ckpt` for model parameter, while other files are used to process data, train and test the model. Note that `violence_predict.ckpt` needs `git lfs` to manage for being too large.

## Call Instance

Here is a call instance for the classify.py interface.

```python
from torchvision import transforms
from PIL import Image
from classify import ViolenceClass


def prepare_images(directory):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # load image from given folder
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            image = transform(image)
            images.append(image)
    images_tensor = torch.stack(images)
    return images_tensor


if __name__ == '__main__':
    model_path = 'violence_predict.ckpt'
    classifier = ViolenceClass(model_path)
    directory = 'images_to_predict/'
    image_tensors = prepare_images(directory)
    predictions = classifier.classify(image_tensor)
    print("Predicted Classes:", predictions)

```

