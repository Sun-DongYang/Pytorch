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

# 训练与验证网络（所有层都参加训练）
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    # 保存网络训练最好的权重
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每训练一个epoch，测试一下网络模型的准确率
        for phase in ['train', 'val']:

            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                # 学习率更新方式
                scheduler.step()
                print (scheduler.get_lr())
                #  调用模型训练
                model.train(True)
                # 依次获取所有图像，参与模型训练或测试
                for data in dataloaders[phase]:
                    # 获取输入
                    inputs, labels = data
                    判断是否使用gpu
                    if use_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    inputs, labels = Variable(inputs), Variable(labels)

                    # 梯度清零
                    optimizer.zero_grad()

                    # 网络前向运行
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    # 计算Loss值
                    loss = criterion(outputs, labels)
                    # 反传梯度
                    loss.backward()
                    # 更新权重
                    optimizer.step()
                    # 计算一个epoch的loss值和准确率
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            else:
                # 定义不保存梯度
                with torch.no_grad():
                    # 调用模型测试
                    model.eval()
                    # 依次获取所有图像，参与模型训练或测试
                    for data in dataloaders[phase]:
                        # 获取输入
                        inputs, labels = data
                        # 判断是否使用gpu
                        if use_gpu:
                            inputs = inputs.cuda()
                            labels = labels.cuda()

                        inputs, labels = Variable(inputs), Variable(labels)

                        # 梯度清零
                        optimizer.zero_grad()

                        # 网络前向运行
                        outputs = model(inputs)
                        _, preds = torch.max(outputs.data, 1)
                        # 计算Loss值
                        loss = criterion(outputs, labels)
                        # 计算一个epoch的loss值和准确率
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
            # 计算Loss和准确率的均值
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 保存测试阶段，准确率最高的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
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
    if use_gpu:
        model_ft = model_ft.cuda()

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

