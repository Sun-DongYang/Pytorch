import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.misc


# 导入数据
def get_image_info(image_dir):
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    image_info = Image.open(image_dir).convert('RGB')
    # 数据预处理方法
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_info = image_transform(image_info)
    image_info = image_info.unsqueeze(0)
    return image_info

# 获取第k层的特征图
def get_k_layer_feature_map(feature_extractor, k, x):
    with torch.no_grad():
        for index,layer in enumerate(feature_extractor):
            x = layer(x)
            if k == index:
                return x

#  可视化特征图
def show_feature_map(feature_map):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num+1):
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[0], cmap='gray')
        plt.axis('off')
        scipy.misc.imsave(str(index)+".png", feature_map[0])
    plt.show()




if __name__ ==  '__main__':
    # 初始化图像的路径
    image_dir = r"husky.png"
    # 定义提取第几层的feature map
    k = 1
    # 导入Pytorch封装的AlexNet网络模型
    model = models.alexnet(pretrained=True)
    # 是否使用gpu运算
    use_gpu = torch.cuda.is_available()
    # 读取图像信息
    image_info = get_image_info(image_dir)
    # 判断是否使用gpu
    if use_gpu:
        model = model.cuda()
        image_info = image_info.cuda()
    # alexnet只有features部分有特征图
    # classifier部分的feature map是向量
    feature_extractor = model.features
    feature_map = get_k_layer_feature_map(feature_extractor, k, image_info)
    show_feature_map(feature_map)


    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(image_info)
    # plt.show()
