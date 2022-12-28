from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import math

# resolution size 32x32 -> 28x28 -> 14x14 -> 10x10 -> 5x5

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_3x3_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )



class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Inverted_residual_block(nn.Module):
    def __init__(self, in_channel, bottleneck_list) : # bottleneck_list [t, c , n ,s]
        super(Inverted_residual_block, self).__init__()
        self.block = bottleneck_list
        hidden_dim = int(in_channel * bottleneck_list[0])

        self.use_res_connect = bottleneck_list[-1] == 1 and in_channel == bottleneck_list[1]

        '''
        Depthwise Separable Convolutions are a key building block for many efficient neural network architectures
        [27, 28, 20] and we use them in the present work as well.
        The basic idea is to replace a full convolutional operator with a factorized version that splits convolution into
        two separate layers. The first layer is called a depthwise
        convolution, it performs lightweight filtering by applying a single convolutional filter per input channel. The
        second layer is a 1 × 1 convolution, called a pointwise
        convolution, which is responsible for building new features through computing linear combinations of the input channel
        
        '''
        if bottleneck_list[0] == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, bottleneck_list[-1], 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, bottleneck_list[1], 1, 1, 0, bias=False),
                nn.BatchNorm2d(bottleneck_list[1]),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(in_channels = in_channel, out_channels=hidden_dim, kernel_size=1, stride=1, padding= 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(in_channels = hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=bottleneck_list[-1], padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(in_channels =hidden_dim, out_channels=bottleneck_list[1], kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(bottleneck_list[1]),
            )


    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,n_classes,input_size=224, width_mult = 1.0):
        super(MobileNetV2, self).__init__()
        # [t, c , n ,s]
        input_channel =int(32 *width_mult)
        self.last_channel = int(1280 *width_mult)

        bottleneck_list = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        block = Inverted_residual_block
        self.mbv2_features = [conv_bn(3, input_channel, 2)]

        for t,c,n,s in bottleneck_list:
            for i in range(n):
                if i == 0:
                    self.mbv2_features.append(block(input_channel, [t,int(c*width_mult),n,s]))
                else:
                    self.mbv2_features.append(block(input_channel, [t,int(c*width_mult),n,s]))
                input_channel = int(c *width_mult)

        self.mbv2_features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.mbv2_features = nn.Sequential(*self.mbv2_features)
        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_classes)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.mbv2_features(x)
        # print('mbv2_features : ', x.size())
        x = x.mean(3).mean(2)

        # print('x_data : ',x.size())

        logits = self.classifier(x)

        # probs = torch.softmax(logits, dim=1)
        return logits

