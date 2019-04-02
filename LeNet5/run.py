from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 读取数据
data_train = MNIST('../data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('../data/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))

# num_workers=8 使用多进程加载数据
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

# 初始化网络
net = LeNet5()

# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 定义网络优化方法
optimizer = optim.Adam(net.parameters(), lr=2e-3)

# 定义训练阶段
def train(epoch):
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        # 初始0梯度
        optimizer.zero_grad()
        # 网络前向运行
        output = net(images)
        # 计算网络的损失函数
        loss = criterion(output, labels)

        # 存储每一次的梯度与迭代次数
        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))
        # 反向传播梯度
        loss.backward()
        # 优化更新权重
        optimizer.step()
		    # 保存网络模型结构
    torch.save(net.state_dict(), 'model//' + str(epoch) + '_model.pkl')



def test():
    # 验证阶段
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    # 取消梯度，避免测试阶段out of memory
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            # 计算准确率
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()

        avg_loss /= len(data_test)
        print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))


def train_and_test(epoch):
    train(epoch)
    test()


def main():
    for e in range(1, 16):
        train_and_test(e)


if __name__ == '__main__':
    main()