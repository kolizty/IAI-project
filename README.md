# Introduction

This classify.py defines the ViolenceClass for violence image classification. It uses `model_path` to init the ViolenceClass by loading from saved checkpoint model. It  uses `n*3*224*224` pytorch tensor as input and outputs a length `n` python list as prediction, 0 for non-violence and 1 for violence.



# Call instance

Here is a call instance for the classify.py.

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
    model_path = 'path_to_your_model_checkpoint.ckpt'
    classifier = ViolenceClass(model_path)
    directory = 'path_to_your_image_directory'
    image_tensors = prepare_images(directory)
    predictions = classifier.classify(image_tensor)
    print("Predicted Classes:", predictions)

```

