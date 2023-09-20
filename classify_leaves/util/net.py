# -*- encoding: utf-8 -*-
'''
@File    :   net.py
@Time    :   2023/09/18 10:11:32
@Author  :   orCate 
@Version :   1.0
@Contact :   8631143542@qq.com
'''

# here put the import lib
import torch
from torch import nn
from torch.nn import functional as f

def Conv(inp_channels, out_channels, kernel_size, padding, stride, isReLU=True, isBN=True):
    if isReLU and isBN:
        ret = nn.Sequential(nn.Conv2d(inp_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride), nn.BatchNorm2d(out_channels), nn.LeakyReLU())
    elif isReLU:
        ret = nn.Sequential(nn.Conv2d(inp_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride), nn.LeakyReLU())
    elif isBN:
        ret = nn.Sequential(nn.Conv2d(inp_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride), nn.BatchNorm2d(out_channels))
    else:
        ret = nn.Sequential(nn.Conv2d(inp_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride))
    return ret
        

class Inception(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(Inception, self).__init__()
        self.forward_channel1 = nn.Conv2d(inp_channels, out_channels // 8 + out_channels // 16, kernel_size=1)

        self.forward_channel2 = nn.Sequential(nn.Conv2d(inp_channels, out_channels // 4, kernel_size=1), 
                                              nn.Conv2d(out_channels // 4, out_channels // 2, kernel_size=3, padding=1))

        self.forward_channel3 = nn.Sequential(nn.Conv2d(inp_channels, out_channels // 8, kernel_size=1), 
                                              nn.Conv2d(out_channels // 8, out_channels // 4, kernel_size=(5, 1), padding=(2, 0)),
                                              nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(1, 5), padding=(0, 2)))
        
        self.forward_channel4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
                                              nn.Conv2d(inp_channels, out_channels// 16, 1))
    
    def forward(self, input):
        out1 = f.leaky_relu(self.forward_channel1(input))
        out2 = f.leaky_relu(self.forward_channel2(input))
        out3 = f.leaky_relu(self.forward_channel3(input))
        out4 = f.leaky_relu(self.forward_channel4(input))

        return torch.cat((out1, out2, out3, out4), dim=1)
    

class ResUnit(nn.Module):
    def __init__(self, inp_channels, out_channels, use_1x1conv=False, strides = 1):
        super(ResUnit, self).__init__()

        self.conv1 = nn.Conv2d(inp_channels, out_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(inp_channels, out_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = f.leaky_relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return f.leaky_relu(Y)

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = Conv(3, 8, kernel_size=3, padding=1, stride=2, isReLU=True, isBN=True)
        self.layer2 = Inception(8, 64)           
        self.layer3 = nn.Sequential(ResUnit(64, 80, True, 2), ResUnit(80, 80), ResUnit(80, 80))
        self.layer4 = nn.Sequential(ResUnit(80, 96, True, 2), ResUnit(96, 96), ResUnit(96, 96))
        self.layer5 = nn.Sequential(ResUnit(96, 112, True, 2), ResUnit(112, 112), ResUnit(112, 112))
        self.layer6 = nn.Sequential(ResUnit(112, 128, True, 2), ResUnit(128, 128), ResUnit(128, 128))
        self.layer7 = Inception(128, 256)
        self.layer8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.layer9 = nn.Sequential(Conv(128, 176, kernel_size=3, padding=1, stride=2, isReLU=True, isBN=True), nn.AdaptiveAvgPool2d((1, 1)),
                                    nn.Flatten())
    
    def forward(self, input):
        out1 = nn.Sequential(self.layer1, self.layer2, self.layer3, 
                             self.layer4, self.layer5, self.layer6, 
                             self.layer7, self.layer8, self.layer9)(input)
    

        return out1

if __name__ == "__main__":
    input = torch.randn((1, 3, 224, 224))
    net = Net()
    net.load_state_dict(torch.load("../checkpoints/best.pth"))
    torch.onnx.export(
        net,
        input,
        "best.onnx",
        export_params=True,
        opset_version=8
    )

