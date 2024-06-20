import torch
from model import ViolenceClassifier


class ViolenceClass:
    def __init__(self, model_path):
        # 加载模型、设置参数等
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ViolenceClassifier().to(self.device)
        self.model = ViolenceClassifier.load_from_checkpoint(model_path)
        self.model.eval()
        print(f"Loading model from {model_path} to {self.device}")

    def misc(self):
        # 其他处理函数
        pass

    def classify(self, imgs: torch.Tensor) -> list:
        # 图像分类
        images_tensor = imgs.to(self.device)
        with torch.no_grad():
            preds = self.model(images_tensor)
            preds = torch.argmax(preds, dim=1).cpu().numpy().tolist()
        return preds
