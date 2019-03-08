from __future__ import print_function, division

import torch
from torchvision import datasets, models, transforms
import os

# 定义数据预处理步骤
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
# 数据路径
data_dir = '../data/hymenoptera_data'
# 调用torchvision.datasets.ImageFolder,实现图像数据的读入和预处理
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
# 调用torch.utils.data.DataLoader，生成Pytorch输入所需的DataLoader格式
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}
# 读取数据集大小
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# 数据类别
class_names = image_datasets['train'].classes

if __name__ == '__main__':
    for x in dataloaders['train']:
        inputs, label = x
        print(inputs)
        print(label)
