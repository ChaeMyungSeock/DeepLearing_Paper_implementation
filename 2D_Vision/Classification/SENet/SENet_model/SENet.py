from torch import nn
import torch.nn.functional as F
from torch import Tensor
import torch

def se_resnetX50(n_classes=10):
    model = SE_ResNext(SE_ResNext_bottleneck, [3, 4, 6, 3], n_classes=n_classes)


    return model

#
#
def se_resnetX101(n_classes=10):
    model = SE_ResNext(SE_ResNext_bottleneck, [3, 4, 23, 3],n_classes=10)



    return model



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SE_ResNext_bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1, ii_downsample = None):
        super(SE_ResNext_bottleneck, self).__init__()
        self.mul = 4
        # stride를 통해 너비와 높이 조정
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # stride = 1, padding = 1이므로, 너비와 높이는 항시 유지됨
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=32)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes*self.mul, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.mul)

        self.downsample = ii_downsample
        self.stride = stride
        self.seblock = SELayer(planes*self.mul)
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
        out = self.seblock(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = F.relu(out)
        return out


class SE_ResNext(nn.Module):
    def __init__(self, resnext_block, num_blocks, n_classes=10, init_weights=True):
        super(SE_ResNext, self).__init__()
        self.init_weights =init_weights

        self.mul = 4
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)


        self.layer1 = self.make_layer(resnext_block, 64, stride=1, num_blocks= num_blocks[0])
        self.layer2 = self.make_layer(resnext_block, 128, stride=2, num_blocks = num_blocks[1])
        self.layer3 = self.make_layer(resnext_block, 256, stride=2, num_blocks = num_blocks[2])
        self.layer4 = self.make_layer(resnext_block, 512, stride=2, num_blocks = num_blocks[3])



        self.fc = nn.Linear(in_features=1000, out_features= n_classes)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*self.mul, out_features=n_classes),
            nn.ReLU6()
        )
        # weights initialization
        if self.init_weights:
            self._initialize_weights()

        # weights initialization function

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def make_layer(self, block, out_planes, stride, num_blocks):
        '''
        (A) The shortcut still
        performs identity mapping, with extra zero entries padded
        for increasing dimensions. This option introduces no extra
        parameter; (B) The projection shortcut in Eqn.(2) is used to
        match dimensions (done by 1×1 convolutions)
        '''
        i_downsample = None

        layers = []


        if stride != 1 or self.in_planes != out_planes * self.mul:  # x와
            i_downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_planes * self.mul, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes * self.mul)
            )

        layers.append(block(self.in_planes, out_planes, stride=stride, ii_downsample = i_downsample))
        self.in_planes = out_planes * self.mul
        for i in range(num_blocks-1):
            layers.append(block(self.in_planes, out_planes))
        # self.in_planes = out_planes
        return nn.Sequential(*layers)



    def forward(self, x):
        # print('x : ', x.size())

        x = torch.relu(self.conv1(x))
        # print('conv1 : ', x.size())


        x = torch.relu(self.maxpool(x))
        # print('maxpool : ', x.size())

        x = self.layer1(x)
        # print('layer1 : ', x.size())

        x = self.layer2(x)
        # print('layer2 : ', x.size())

        x = self.layer3(x)
        # print('layer3 : ', x.size())

        x = self.layer4(x)
        # print('layer4 : ', x.size())

        x = self.avgpool(x)

        x = x.reshape(x.shape[0],-1)
        logits = self.classifier(x)
        # print('classifier : ', logits.size())

        # probs = torch.softmax(logits, dim=1)
        return logits

