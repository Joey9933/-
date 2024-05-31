import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
import time
import os
import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--batchsize", type=int, default=32)
parser.add_argument("--ckp", type=str)
args = parser.parse_args()

CKP_PATH = args.ckp
BATCH_SIZE = args.batchsize
NUM_CLASSES = 200

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 路径设置
DIR = os.getcwd()
data_dir = os.path.join(DIR,"CUB_200_2011\dataset")
ckp_dir = os.path.join(DIR,"checkpoints")

# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 加载数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# 加载模型
print("加载模型中...")
model = torch.load(os.path.join(ckp_dir,CKP_PATH))
model = model.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

def test():# 测试模型
    running_loss = 0
    running_corrects =0
    with torch.no_grad():
        # 遍历数据
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs,labels)

            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        test_loss = running_loss / dataset_sizes['val']
        test_acc = running_corrects.double() / dataset_sizes['val']
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("test", test_loss, test_acc))

    print("执行完毕!")

if __name__=='__main__':
    test()
