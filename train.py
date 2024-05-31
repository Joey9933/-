import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
import time
import os
import copy
import argparse
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=60)
parser.add_argument("--batchsize", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--mode", type=bool, default=True)
args = parser.parse_args()
# LR = 0.01
# NUM_EPOCH = 65
# USE_PRETRAIN = True
USE_PRETRAIN = args.mode
LR = args.lr
NUM_EPOCH = args.epoch
BATCH_SIZE = args.batchsize
NUM_CLASSES = 200

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 路径设置
# data_dir = "path_to_CUB200_dataset"
DIR = os.getcwd()
data_dir = os.path.join(DIR,"CUB_200_2011\dataset")
if not os.path.exists(os.path.join(DIR,f'results_pretrain_{USE_PRETRAIN}')):
    os.mkdir(os.path.join(DIR,f'results_pretrain_{USE_PRETRAIN}'))
    folders = [
        f"./results_pretrain_{USE_PRETRAIN}/logdir",
        f"./results_pretrain_{USE_PRETRAIN}/logdir/train",
        f"./results_pretrain_{USE_PRETRAIN}/logdir/val",
        f"./results_pretrain_{USE_PRETRAIN}/checkpoints",
        f"./results_pretrain_{USE_PRETRAIN}/logs"
    ]
    for folder in folders:
        if not os.path.exists(folder):
            print(f"创建文件夹:{folder}")
            os.makedirs(folder, exist_ok=True)

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
if USE_PRETRAIN:#使用预训练参数，并设置不同的学习率
    model = models.resnet18(pretrained = False)
    model.load_state_dict(torch.load(os.path.join(DIR,'resnet18-5c106cde.pth')),strict=True)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=NUM_CLASSES)

    fc_params_id = list(map(id, model.fc.parameters()))     # 返回的是parameters的 内存地址
    base_params = filter(lambda p: id(p) not in fc_params_id, model.parameters())
    optimizer = optim.SGD([
        {'params': base_params, 'lr': LR*0.1},   # 用更小的lr训练其他层；如果设置为0 ，相当于冻结卷积层，
        {'params': model.fc.parameters(), 'lr': LR}], momentum=0.9)

else:# 加载ResNet-18模型，并随机初始化参数
    model = models.resnet18(weights = None)
    # 修改ResNet-18的输出层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    # 定义参数初始化函数
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
    # 对模型应用初始化
    model.apply(initialize_weights)
    # for m in model.modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         nn.init.xavier_uniform_(m.weight)
    
    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,weight_decay=5e-4)# pretrain使用的
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9,weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

model = model.to(device)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()

# 学习率调度器
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=NUM_EPOCH//3, gamma=0.01)   

# 训练模型
def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    train_writer = SummaryWriter(f"./results_pretrain_{USE_PRETRAIN}/logdir/train")
    val_writer = SummaryWriter(f"./results_pretrain_{USE_PRETRAIN}/logdir/val")

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个epoch有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                # 训练阶段时记录操作
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 训练阶段时反向传播+优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                scheduler.step()
                train_writer.add_scalar("loss", epoch_loss, epoch)
                train_writer.add_scalar("acc", epoch_acc, epoch)
            elif phase == 'val':
                val_writer.add_scalar("loss", epoch_loss, epoch)
                val_writer.add_scalar("acc", epoch_acc, epoch)

            # 深度拷贝模型
            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    train_writer.close()
    val_writer.close()

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model,best_epoch

if __name__=='__main__':
# 开始训练
    model,bestE = train_model(model, criterion, optimizer, lr_scheduler, num_epochs=NUM_EPOCH)
    # model = train_model(model, criterion, optimizer, lr_scheduler, num_epochs=20)
# 保存参数ckp
    torch.save(model, f"./results_pretrain_{USE_PRETRAIN}/checkpoints/best_epoch{bestE}.pth")

