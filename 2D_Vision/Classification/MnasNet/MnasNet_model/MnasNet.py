from torch import nn
import torch.nn.functional as F
from torch import Tensor
import torch

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SepConv(nn.Module):
    def __init__(self,in_channle,out_channel):
        super(SepConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channle, in_channle, kernel_size=3, stride=1, padding=1, groups=in_channle, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channle)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channle, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x

# class Inverted_residual_block(nn.Module):
#     def __init__(self, in_channel, bottleneck_list, mul):  # bottleneck_list [t, c , n ,s]
#         super(Inverted_residual_block, self).__init__()
#         self.block = bottleneck_list
#         hidden_dim = int(in_channel * bottleneck_list[0])
#
#         self.use_res_connect = bottleneck_list[-1] == 1 and in_channel == bottleneck_list[1]
#
#         '''
#         Depthwise Separable Convolutions are a key building block for many efficient neural network architectures
#         [27, 28, 20] and we use them in the present work as well.
#         The basic idea is to replace a full convolutional operator with a factorized version that splits convolution into
#         two separate layers. The first layer is called a depthwise
#         convolution, it performs lightweight filtering by applying a single convolutional filter per input channel. The
#         second layer is a 1 × 1 convolution, called a pointwise
#         convolution, which is responsible for building new features through computing linear combinations of the input channel
#
#         '''
#         if bottleneck_list[0] == 1:
#             self.conv = nn.Sequential(
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, 3, bottleneck_list[-1], 1, groups=hidden_dim, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, bottleneck_list[1], 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(bottleneck_list[1]),
#             )
#         else:
#             self.conv = nn.Sequential(
#                 # pw
#                 nn.Conv2d(in_channels=in_channel, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0,
#                           bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # dw
#                 nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=bottleneck_list[-1],
#                           padding=1, groups=hidden_dim, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(in_channels=hidden_dim, out_channels=bottleneck_list[1], kernel_size=1, stride=1, padding=0,
#                           bias=False),
#                 nn.BatchNorm2d(bottleneck_list[1]),
#             )
#
#     def forward(self, x):
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)


class SE_MB_bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1, mul=4, kernel_size=3, se_ratio=False,ii_downsample = None ):
        super(SE_MB_bottleneck, self).__init__()
        self.mul = mul
        hidden_dim = int(in_planes * self.mul)

        # stride를 통해 너비와 높이 조정
        self.conv1 = nn.Conv2d(in_planes, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.se_ratio = se_ratio
        # stride = 1, padding = 1이므로, 너비와 높이는 항시 유지됨
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False, groups=hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.downsample = ii_downsample
        self.stride = stride
        self.seblock = SELayer(planes)
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
        if self.se_ratio:
            out = self.seblock(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = F.relu(out)
        return out


class MnasNet(nn.Module):
    def __init__(self, MBConvblock, num_blocks, n_classes=10):
        super(MnasNet, self).__init__()
        self.mul = 4
        self.in_planes = 16
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.seqconv = SepConv(in_channle=32, out_channel=16)

        self.layer1 = self.make_layer(MBConvblock, 16, 24, stride=2, mul=6, kernel_size=3, num_blocks= num_blocks[0], se_ratio=False)
        self.layer2 = self.make_layer(MBConvblock, 24, 40, stride=2, mul=3, kernel_size=5, num_blocks = num_blocks[1], se_ratio=True)
        self.layer3 = self.make_layer(MBConvblock, 40, 80, stride=2, mul=6, kernel_size=3, num_blocks = num_blocks[2], se_ratio=False)
        self.layer4 = self.make_layer(MBConvblock, 80, 112, stride=1, mul=6, kernel_size=3, num_blocks = num_blocks[3], se_ratio=True)
        self.layer5 = self.make_layer(MBConvblock, 112, 160, stride=2, mul=6, kernel_size=5, num_blocks = num_blocks[3], se_ratio=True)
        self.layer6 = self.make_layer(MBConvblock, 160, 320, stride=1, mul=6, kernel_size=3, num_blocks = num_blocks[3], se_ratio=False)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=320, out_features= 1024),
            nn.Linear(in_features=1024, out_features=n_classes)

        )


    def make_layer(self, block, in_planes,out_planes, stride, num_blocks, kernel_size, mul, se_ratio=False):
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
                nn.Conv2d(self.in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

        layers.append(block(self.in_planes, out_planes, kernel_size=kernel_size, mul = mul, stride=stride, ii_downsample = i_downsample, se_ratio=se_ratio))
        self.in_planes = out_planes
        for i in range(num_blocks-1):
            layers.append(block(self.in_planes, out_planes))
        # self.in_planes = out_planes
        return nn.Sequential(*layers)



    def forward(self, x):

        x = torch.relu(self.conv1(x))
        print('conv1 : ', x.size())

        x = self.seqconv(x)
        print('seqconv : ', x.size())


        x = self.layer1(x)
        print('MBConv6 (k3x3) : ', x.size())

        x = self.layer2(x)
        print('MBConv3 (k5x5) SE : ', x.size())
        #
        x = self.layer3(x)
        print('MBConv6 (k3x3) SE : ', x.size())

        x = self.layer4(x)
        print('layer4 : ', x.size())

        x = self.layer5(x)
        print('layer5 : ', x.size())

        x = self.layer6(x)
        print('layer6 : ', x.size())

        x = self.avgpool(x)
        print('avg : ', x.size())

        #
        x = x.reshape(x.shape[0],-1)
        logits = self.classifier(x)
        print('classifier : ', logits.size())


        probs = torch.softmax(logits, dim=1)
        return logits, probs
