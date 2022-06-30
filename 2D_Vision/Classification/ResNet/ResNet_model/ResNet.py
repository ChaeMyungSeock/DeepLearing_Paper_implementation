from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.mul = 1
        # stride를 통해 너비와 높이 조정
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        # stride = 1, padding = 1이므로, 너비와 높이는 항시 유지됨
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # x를 그대로 더해주기 위함
        self.shortcut = nn.Sequential()
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

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # 필요에 따라 layer를 Skip
        out = F.relu(out)
        return out


class Res_bottleneck(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Res_bottleneck, self).__init__()
        self.mul = 4
        # stride를 통해 너비와 높이 조정
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # stride = 1, padding = 1이므로, 너비와 높이는 항시 유지됨
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes*self.mul, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.mul)

        # x를 그대로 더해주기 위함
        self.shortcut = nn.Sequential()
        '''
        (A) The shortcut still
        performs identity mapping, with extra zero entries padded
        for increasing dimensions. This option introduces no extra
        parameter; (B) The projection shortcut in Eqn.(2) is used to
        match dimensions (done by 1×1 convolutions)
        '''

        # 만약 size가 안맞아 합연산이 불가하다면, 연산 가능하도록 모양을 맞춰줌
        if stride != 1 or in_planes != planes*self.mul:  # x와
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes*self.mul, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*self.mul)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)  # 필요에 따라 layer를 Skip
        out = F.relu(out)
        return out
