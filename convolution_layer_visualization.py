from PIL import Image
from torchvision import transforms as T
import torch
from torch import nn as nn
import numpy as np
import torchvision

def conv2d(img_path, kernel=1, stride=1, padding=0):
    img = Image.open(img_path).convert("RGB")
    img = T.ToTensor()(img)
    img = torch.unsqueeze(img, 0)
    conv = nn.Conv2d(3, 6, kernel_size=kernel, stride=stride, padding=padding)
    # m = nn.MaxPool2d(3, stride=1)
    img = conv(img)
    print(img.size())
    i = img[0,1].view(1,img.size()[2],img.size()[3])
    return T.ToPILImage()(i.to('cpu'))
    
def conv_transpose_2d(img_path, kernel=1, stride=1, padding=0):
    img = Image.open(img_path).convert("RGB")
    img = T.ToTensor()(img)
    img = torch.unsqueeze(img, 0)
    conv = nn.ConvTranspose2d(3, 6, kernel_size=kernel, stride=stride, padding=padding)
    img = conv(img)
    print(img.size())
    i = img[0,1].view(1,img.size()[2],img.size()[3])
    return T.ToPILImage()(i.to('cpu'))


conv2d('c:/users/armin/desktop/2.jpg')