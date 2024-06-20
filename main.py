from PIL import Image
from torchvision import transforms
from classify import ViolenceClass

if __name__ == '__main__':
    # test classify interface
    classifier = ViolenceClass('violence_predict.ckpt')
    img_path = 'dataset/test_original/1_2125.jpg'
    images = Image.open(img_path)
    transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    images = transforms(images)
    print(classifier.classify(images.unsqueeze(0)))
