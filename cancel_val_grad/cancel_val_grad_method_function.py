from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import copy

# 是否使用gpu运算
use_gpu = torch.cuda.is_available()
# 数据预处理
data_transforms = {
    'train': transforms.Compose([
        # 随机在图像上裁剪出224*224大小的图像
        transforms.RandomResizedCrop(224),
        # 将图像随机翻转
        transforms.RandomHorizontalFlip(),
        # 将图像数据,转换为网络训练所需的tensor向量
        transforms.ToTensor(),
        # 图像归一化处理
        # 个人理解,前面是3个通道的均值,后面是3个通道的方差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 读取数据
# 这种数据读取方法,需要有train和val两个文件夹，
# 每个文件夹下一类图像存在一个文件夹下
data_dir = '../data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

# 读取数据集大小
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# 数据类别
class_names = image_datasets['train'].classes

# 定义训练阶段
def train(model, criterion, optimizer, scheduler, phase='train'):
    # 保存训练一个epoch的Loss值与准确率
    running_loss = 0.0
    running_corrects = 0
    # 更新学习率
    scheduler.step()
    # 指定模型训练
    model.train()
    for data in dataloaders[phase]:
        inputs, labels = data

        # 判断是否使用gpu
        # if use_gpu:
        #     inputs = inputs.cuda()
        #     labels = labels.cuda()

        inputs = Variable(inputs)
        labels = Variable(labels)

        # 清除梯度
        optimizer.zero_grad()
        # 模型前向运行
        outputs = model(inputs)
        # 计算预测结果
        _, preds = torch.max(outputs.data, 1)
        # 计算Loss值
        loss = criterion(outputs, labels)
        # 反传loss
        loss.backward()
        # 更新模型权重
        optimizer.step()
        # 统计Loss值
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = float(running_corrects) / dataset_sizes[phase]
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    return model


# 定义验证阶段
def val(model, criterion, phase='val'):
    # 模型验证
    model.eval()
    # 指定不保存梯度
    with torch.no_grad():
        # 统计Loss值与准确率
        running_loss = 0.0
        running_corrects = 0

        for data in dataloaders[phase]:
            inputs, labels = data
            # 判断是否使用gpu
            # if use_gpu:
            #     inputs = inputs.cuda()
            #     labels = labels.cuda()

            inputs = Variable(inputs)
            labels = Variable(labels)
            # 模型前向运行
            outputs = model(inputs)
            # 获取预测结果
            _, preds = torch.max(outputs.data, 1)
            # 计算Loss值
            loss = criterion(outputs, labels)
            # 统计Loss值和准确率
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = float(running_corrects) / dataset_sizes[phase]
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        return epoch_acc

# 定义网络训练（中间夹杂着验证）
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    # 保存网络训练最好的权重
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model = train(model, criterion, optimizer, scheduler)
        acc = val(model, criterion)

        # 保存测试阶段，准确率最高的模型
        if  acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # 网络导入最好的网络权重
    model.load_state_dict(best_model_wts)
    return model


# 微调网络
if __name__ ==  '__main__':

    # 导入Pytorch中自带的resnet18网络模型
    model_ft = models.resnet18(pretrained=True)
    # 将网络模型的各层的梯度更新置为False
    for param in model_ft.parameters():
        param.requires_grad = False

    # 修改网络模型的最后一个全连接层
    # 获取最后一个全连接层的输入通道数
    num_ftrs = model_ft.fc.in_features
    # 修改最后一个全连接层的的输出数为2
    model_ft.fc = nn.Linear(num_ftrs, 2)
    # 是否使用gpu
    # if use_gpu:
    #     model_ft = model_ft.cuda()

    # 定义网络模型的损失函数
    criterion = nn.CrossEntropyLoss()

    # 只训练最后一个层
    # 采用随机梯度下降的方式，来优化网络模型
    optimizer_ft = torch.optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)

    # 定义学习率的更新方式，每5个epoch修改一次学习率
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.1)
    # 训练网络模型
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10)
    # 存储网络模型的权重
    torch.save(model_ft.state_dict(),"model_only_fc.pkl")
