from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model import ViolenceClassifier
from dataset import CustomDataModule

gpu_id = [0]
batch_size = 128
log_name = "resnet18_pretrain"

data_module = CustomDataModule(batch_size=batch_size)
ckpt_root = "train_logs/"
ckpt_version = 4
ckpt_path = (ckpt_root + "resnet18_pretrain_test/version_" + str(ckpt_version) +
             "/checkpoints/resnet18_pretrain_test-epoch=29-val_loss=0.04.ckpt")
logger = TensorBoardLogger("test_logs", name=log_name)

model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
trainer = Trainer(accelerator='gpu', devices=gpu_id)

if __name__ == '__main__':
    trainer.test(model, data_module)
