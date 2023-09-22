"""

@Author: So What
@Time: 2023/9/21 20:07
@File: practice.py

"""
import torch
import torchvision
from torch import nn
from torch.nn.modules import transformer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from PIL import Image
from model import *

trans_resize=transforms.Resize((1024,1024))
tensor_trans=transforms.ToTensor()
#准备数据集
class MyData(Dataset):
    def __init__(self,root_dir,label_dir):   #初始化类，初始化函数在类创建实例的时候会运行的函数
        dataset=[]
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path=os.listdir(self.path) #获得指定路径下所有图片地址，组成列表
        for img in self.img_path:
            if self.label_dir=='sunny':
                dataset.append((img,0))
            else:
                dataset.append((img,1))
        self.dataset=dataset
    def __getitem__(self, idx):  #idx是为了获取对应的图片
        pic,label=self.dataset[idx]
        pic=Image.open(self.path+'\\'+pic)
        pic=trans_resize(pic)
        pic=tensor_trans(pic)
        return pic,label
    def __len__(self):
        return len(self.img_path)

root_dir="dataset/train"
sunny_label_dir="sunny"
cloudy_label_dir="cloudy"
sunny_dataset=MyData(root_dir,sunny_label_dir)
cloudy_dataset=MyData(root_dir,cloudy_label_dir)
train_dataset=sunny_dataset+cloudy_dataset  #将两个数据集拼接
train_data_size = len(train_dataset)

root_dir2='dataset/test'
sunny_test='sunny'
cloudy_test='cloudy'
sunny_test_dataset=MyData(root_dir2,sunny_test)
cloudy_test_dataset=MyData(root_dir2,cloudy_test)
test_dataset=sunny_test_dataset+cloudy_test_dataset
# print(type(train_dataset))
train_dataloader = DataLoader(train_dataset, batch_size=2,shuffle=True) #训练数据集
test_dataloader = DataLoader(test_dataset, batch_size=2,shuffle=True)
model=myModel()

#损失函数
loss_fn=nn.CrossEntropyLoss()

#优化器
learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)


for data in train_dataloader:
    imgs, targets = data
    outputs = model(imgs)
    print(outputs)
    print(targets)
    loss = loss_fn(outputs, targets) #查看输出与真实的target距
    print(loss)
    # # 优化器优化模型
    # optimizer.zero_grad() #梯度清0
    # loss.backward()
    # optimizer.step()
    # total_train_step = total_train_step+1
    # print("训练次数:{},Loss:{}".format(total_train_step, loss.item()))




