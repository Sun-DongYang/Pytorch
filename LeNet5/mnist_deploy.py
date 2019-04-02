from lenet import LeNet5
import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 读取数据
data_test = MNIST('../data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

# num_workers=8 使用多进程加载数据
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

# 初始化网络
net = LeNet5()
net.load_state_dict(torch.load(r'model/1_model.pkl'))

def deploy():
    # 验证阶段
    net.eval()
    total_correct = 0
    # 取消测试阶段的梯度，避免out of memory
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            output = net(images)
            # 计算准确率
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
        print('Accuracy: %f' % ( float(total_correct) / len(data_test)))


def main():
    deploy()


if __name__ == '__main__':
    main()
