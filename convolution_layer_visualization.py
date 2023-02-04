from PIL import Image
from torchvision import transforms as T
import torch
from torch import nn as nn
import numpy as np
import torchvision

def main_function(img_path):
    img = Image.open('c:/users/armin/desktop/air.jpg').convert("RGB")
    img = T.ToTensor()(img)
    img = torch.unsqueeze(img, 0)
    conv = nn.Conv2d(3, 6, 3, stride=1)
    m = nn.MaxPool2d(3, stride=1)
    img = conv(img)
    img = m(img)
    print(img.size())
    i = img[0,1].view(1,img.size()[2],img.size()[3])
    return T.ToPILImage()(i.to('cpu'))
    

main_function('c:/users/armin/desktop/air.jpg')