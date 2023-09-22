"""

@Author: So What
@Time: 2023/9/22 9:11
@File: test.py

"""
import torch
import torchvision
from PIL import Image
from model import *  # 方式1加载保存好的模型需要copy model class或者导入model文件

# image_path="../imgs/dog.png"
# image_path = "D:\\Python_Projects\\Learning_PyTorch\\MyProjects\\dataset\\test\\sunny\\IMG_20230806_180603.jpg"
# image_path = "D:\\Python_Projects\\Learning_PyTorch\\MyProjects\\dataset\\test\\sunny\\IMG_20230825_125904.jpg"
# image_path = "D:\\Python_Projects\\Learning_PyTorch\\MyProjects\\dataset\\test\\cloudy\\IMG_20230822_160241.jpg"
# image_path = "D:\\Python_Projects\\Learning_PyTorch\\MyProjects\\dataset\\test\\cloudy\\IMG_20230807_174150.jpg"
# image_path='dataset/test/sunny/IMG_20230709_143345.jpg'
# image_path='dataset/test/cloudy/IMG_20230807_174150.jpg'
image_path='./dataset/test/cloudy/IMG_20230807_174150.jpg/'
# image_path
image = Image.open(image_path)
# print(image)  # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=202x213 at 0x27F1DAB0F70>
image = image.convert('RGB')  # png格式有四个通道：RGB+Alph透明通道 调用convert保留其RGB通道

# model要求输入只能是32*32
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((1024, 1024)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)  # torch.Size([3, 32, 32])

# 载入train好的模型（数据、参数及模型结构）
# model=torch.load('model_0.pth')#方式1导入
model = torch.load('../model2.pth')  # 方式1导入

image = torch.reshape(image, (1, 3, 1024, 1024))  # 模型输入需要是四维数据
# image = image.cuda()  # 若保存的模型是用gpu训练的，则需要将输入数据同样写为在gpu训练形式
model.eval()  # 进入测试状态
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax())
print(output.argmax(1))