"""

@Author: So What
@Time: 2023/9/21 18:41
@File: model.py

"""
# 为了规范，一般会把该部分单独放在一个文件中，然后引入
import torch
from torch import nn


# 搭建神经网络
class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()  #对父类的初始化
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, 2, 2, 0),  # in_channels=3,out_channels=8,kernel_size=2,stride=2,padding=0
            nn.MaxPool2d(4),
            nn.Conv2d(8, 8, 4, 4, 0),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.Linear(256,64),
            nn.Linear(64,10),
            nn.Linear(10, 2)
        )
    def forward(self, x):
        x = self.model(x)
        return x


# 测试网络模型    给定一个确定的输入尺寸，查看输出尺寸是否为我们想要的
if __name__ == '__main__':  # main
    model = myModel()
    input = torch.ones((2, 3, 1024, 1024))
    output = model(input)
    print(output.shape)
    print(output)
   # 输出为torch.Size([64, 10])，返回64行数据，每一行数据有10个数据
