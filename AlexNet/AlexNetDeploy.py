from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os

# 是否使用gpu运算
use_gpu = torch.cuda.is_available()
# 导入数据
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

# 读取数据要用的函数
# 这个以后可以自己写
data_dir = '../data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
# 读取数据集大小
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}



if __name__ ==  '__main__':
    # 导入Pytorch封装的AlexNet网络模型
    model = models.alexnet(pretrained=True)
    # 获取最后一个全连接层的输入通道数
    num_input = model.classifier[6].in_features
    # 获取全连接层的网络结构
    feature_model = list(model.classifier.children())
    # 去掉原来的最后一层
    feature_model.pop()
    # 添加上适用于自己数据集的全连接层
    feature_model.append(nn.Linear(num_input, 2))
    # 仿照这里的方法，可以修改网络的结构，不仅可以修改最后一个全连接层
    # 还可以为网络添加新的层
    # 重新生成网络的后半部分
    model.classifier = nn.Sequential(*feature_model)
    if use_gpu:
        model = model.cuda()
    model.load_state_dict(torch.load('model_AlexNet.pkl'))

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        # 判断是否使用gpu
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        # 网络的前一部分
        information = model.features(inputs)
        # 拉成一维向量
        information = information.view(information.size(0), -1)
        # 提取倒数第二层的特征信息[0:5]
        # 提取倒数第三层的特征信息[0:4]
        # 提取网络的前半部分的特征信息model.features[0:n](information)
        information = model.classifier[0:5](information)
        print (information.shape)
