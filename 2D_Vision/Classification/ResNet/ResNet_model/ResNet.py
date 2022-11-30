from torch import nn
import torch.nn.functional as F
from torch import Tensor
import torch
from typing import Any, Callable, List, Optional, Type, Union

def resnet18(n_classes):
    model = ResNet(BasicBlock,  [2, 2, 2, 2], n_classes=n_classes)
    return model


def resnet34(n_classes):
    model = ResNet(BasicBlock, [3, 4, 6, 3], n_classes=n_classes)

    return model


#
def resnet50(n_classes):
    model = ResNet(Res_bottleneck, [3, 4, 6, 3], n_classes=n_classes)
    return model


#
def resnet101(n_classes):
    model = ResNet(Res_bottleneck, [3, 4, 23, 3], n_classes=n_classes)
    return model

#
def resnet152(n_classes):
    model = ResNet(Res_bottleneck, [3, 8, 36, 3], n_classes=n_classes)
    return model




class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample = None):
        super(BasicBlock, self).__init__()
        # stride를 통해 너비와 높이 조정
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        # stride = 1, padding = 1이므로, 너비와 높이는 항시 유지됨
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample

        '''
        (A) The shortcut still
        performs identity mapping, with extra zero entries padded
        for increasing dimensions. This option introduces no extra
        parameter; (B) The projection shortcut in Eqn.(2) is used to
        match dimensions (done by 1×1 convolutions)
        '''

        # 만약 size가 안맞아 합연산이 불가하다면, 연산 가능하도록 모양을 맞춰줌
        if stride != 1:  # x와
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x : Tensor) -> Tensor:
        identity = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out


class Res_bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_planes, planes, stride=1, downsample = None):
        super(Res_bottleneck, self).__init__()
        # stride를 통해 너비와 높이 조정
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # stride = 1, padding = 1이므로, 너비와 높이는 항시 유지됨
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.downsample = downsample
        self.stride = stride
        # x를 그대로 더해주기 위함
        '''
        (A) The shortcut still
        performs identity mapping, with extra zero entries padded
        for increasing dimensions. This option introduces no extra
        parameter; (B) The projection shortcut in Eqn.(2) is used to
        match dimensions (done by 1×1 convolutions)
        '''
        #
        # # 만약 size가 안맞아 합연산이 불가하다면, 연산 가능하도록 모양을 맞춰줌
        # if stride != 1 or in_planes != planes*self.mul:  # x와
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_planes, planes*self.mul, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(planes*self.mul)
        #     )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,
        block: Type[Union[BasicBlock, Res_bottleneck]],
        num_blocks, n_classes=10) -> None :
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)


        self.layer1 = self.make_layer(block, 64, stride=1, num_blocks= num_blocks[0])
        self.layer2 = self.make_layer(block, 128, stride=2, num_blocks = num_blocks[1])
        self.layer3 = self.make_layer(block, 256, stride=2, num_blocks = num_blocks[2])
        self.layer4 = self.make_layer(block, 512, stride=2, num_blocks = num_blocks[3])



        # self.fc = nn.Linear(in_features=1000, out_features= n_classes)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*block.expansion, out_features=n_classes),
            nn.ReLU6()
        )

    def make_layer(self, block: Type[Union[BasicBlock,Res_bottleneck ]], out_planes, stride, num_blocks):
        '''
        (A) The shortcut still
        performs identity mapping, with extra zero entries padded
        for increasing dimensions. This option introduces no extra
        parameter; (B) The projection shortcut in Eqn.(2) is used to
        match dimensions (done by 1×1 convolutions)
        '''
        i_downsample = None

        layers = []


        if stride != 1 or self.in_planes != out_planes * block.expansion:  # x와
            i_downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * block.expansion)
            )

        layers.append(block(self.in_planes, out_planes, stride=stride, downsample = i_downsample))
        self.in_planes = out_planes * block.expansion
        for i in range(num_blocks-1):
            layers.append(block(self.in_planes, out_planes))
        # self.in_planes = out_planes
        return nn.Sequential(*layers)



    def forward(self, x):
        x = torch.relu(self.conv1(x))

        x = torch.relu(self.maxpool(x))

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.reshape(x.shape[0],-1)
        logits = self.classifier(x)

        # probs = torch.softmax(logits, dim=1)
        return logits