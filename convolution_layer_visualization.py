from PIL import Image
from torchvision import transforms as T
import torch
from torch import nn as nn
import numpy as np
import torchvision

def conv2d(img_path, kernel=3, stride=2, padding=2):
    img = Image.open(img_path).convert("RGB")
    img = T.ToTensor()(img)
    # img = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.100, 0.100, 0.100])(img)
    
    img = torch.unsqueeze(img, 0)
    conv = nn.Conv2d(3, 6, kernel_size=kernel, stride=stride, padding=padding, dilation=2)
    conv2 = nn.Conv2d(6, 9, kernel_size=kernel, stride=stride, padding=padding, dilation=2)
    # m = nn.MaxPool2d(3, stride=1)
    img = conv2(conv(img))
    i = img[0,1].view(1,img.size()[2],img.size()[3])
    tns = (i.to('cpu')).detach().numpy()
    s = nn.Softmax(dim=2)
    return s(i)
    # return tns
    return T.ToPILImage()(i.to('cpu'))
    
def conv_transpose_2d(img_path, kernel=3, stride=1, padding=0):
    img = Image.open(img_path).convert("RGB")
    img = T.ToTensor()(img)
    img = torch.unsqueeze(img, 0)
    conv = nn.ConvTranspose2d(3, 6, kernel_size=kernel, stride=stride, padding=padding)
    img = conv(img)
    print(img.size())
    i = img[0,1].view(1,img.size()[2],img.size()[3])
    return T.ToPILImage()(i.to('cpu'))


tensor = conv2d('c:/users/armin/desktop/4.jpg')
v = tensor.mul(255).byte()
T.ToPILImage()(v)
tensor
# v.min()
#%%
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.conv3 = nn.Conv2d(20, 20, 5)
        self.seq = nn.ModuleList([nn.Conv2d(1, 20, 5),nn.Conv2d(20, 20, 5)])
    def extra_repr(self):
        return "Helllllllllllllo"
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
    
m = Model()
# m.register_buffer('running_mean', nn.Conv2d(20, 20, 5))
m
# for i,j in enumerate(m.modules()):
#     print(j)
# m.get_buffer('conv1')
#%%
def round_to_step(x, step):
    return np.round(x / step) * step
img = Image.open('c:/users/armin/desktop/2.jpg')
npp = np.asarray(img)
npp2 = npp
npp = round_to_step(npp,50)
Image.fromarray(np.uint8(npp))
npp2