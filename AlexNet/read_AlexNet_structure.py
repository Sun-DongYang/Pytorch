import torchvision.models as models

# 导入Pytorch所封装的AlexNet模型
model = models.alexnet(pretrained=True)

# model.features存储的是网络模型的前半部分
# 即卷积层与池化层部分
model_features = list(model.features.children())
for feature in model_features:
    print (feature)
# model.classifier存储的是模型的后半部分
# 即全连接层部分
model_classifier = list(model.classifier.children())
for classifier in model_classifier:
    print (classifier)

