from torch import nn
import torch.nn.functional as F
import torch

# resolution size 32x32 -> 28x28 -> 14x14 -> 10x10 -> 5x5
class depthwise_conv_block(nn.Module):
    def __init__(self, in_channel,out_channel, kernels_per_layer=1, stride=1,padding=1):
        super(depthwise_conv_block, self).__init__()
        self.depthwise = nn.Conv2d(in_channels = in_channel,out_channels= in_channel * kernels_per_layer, kernel_size=3, padding=padding, stride=stride,groups=in_channel)
        self.bn1 = nn.BatchNorm2d(in_channel * kernels_per_layer)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d( in_channels= in_channel * kernels_per_layer, out_channels=out_channel, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class MobileNet(nn.Module):
    def __init__(self,n_classes,alpha):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=int(32*alpha), kernel_size=3, stride=2,padding=1)

        self.depthwise_conv1 = depthwise_conv_block(in_channel=int(32*alpha),out_channel=int(64*alpha), stride=1)
        self.depthwise_conv2 = depthwise_conv_block(in_channel=int(64*alpha),out_channel=int(128*alpha), stride=2)

        self.depthwise_conv3 = depthwise_conv_block(in_channel=int(128*alpha),out_channel=int(128*alpha),kernels_per_layer=1, stride=1)
        self.depthwise_conv4 = depthwise_conv_block(in_channel=int(128*alpha),out_channel=int(256*alpha),kernels_per_layer=1, stride=2)

        self.depthwise_conv5 = depthwise_conv_block(in_channel=int(256*alpha), out_channel=int(256*alpha), kernels_per_layer=1, stride=1)
        self.depthwise_conv6 = depthwise_conv_block(in_channel=int(256*alpha), out_channel=int(512*alpha), kernels_per_layer=1, stride=2)


        self.depthwise_conv7 = depthwise_conv_block(in_channel=int(512*alpha), out_channel=int(512*alpha), kernels_per_layer=1, stride=1) # x 5

        self.depthwise_conv8 = depthwise_conv_block(in_channel=int(512*alpha), out_channel=int(1024*alpha), kernels_per_layer=1, stride=2)

        self.depthwise_conv9 = depthwise_conv_block(in_channel=int(1024*alpha), out_channel=int(1024*alpha), kernels_per_layer=1, stride=1, padding=1)

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_features=int(1024*alpha), out_features= n_classes)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=int(1024*alpha), out_features= n_classes)
        )





    def forward(self, x):

        x = torch.relu(self.conv1(x))
        # print('conv1 : ', x.size())
        x = self.depthwise_conv1(x)
        x = self.depthwise_conv2(x)
        # print('depthwise_conv2 : ', x.size())

        x = self.depthwise_conv3(x)
        x = self.depthwise_conv4(x)
        # print('depthwise_conv4 : ', x.size())

        x = self.depthwise_conv5(x)
        x = self.depthwise_conv6(x)
        # print('depthwise_conv6 : ', x.size())

        for i in range(5):
            x = self.depthwise_conv7(x)
        # print('depthwise_conv7 : ', x.size())

        x = self.depthwise_conv8(x)
        # print('depthwise_conv8 : ', x.size())

        x = self.depthwise_conv9(x)
        # print('depthwise_conv9 : ', x.size())

        x = torch.relu(self.avg(x))
        # print('fc : ', x.size())


        x = x.view(-1,1024)
        logits = self.classifier(x)
        # print('classifier : ', logits.size())

        probs = torch.softmax(logits, dim=1)
        return probs

