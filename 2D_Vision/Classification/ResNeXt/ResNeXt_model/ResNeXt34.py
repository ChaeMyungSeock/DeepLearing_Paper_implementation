from torch import nn
import torch.nn.functional as F
import torch



class ResNeXt34(nn.Module):
    def __init__(self, block, num_blocks, n_classes=10):
        super(ResNeXt34, self).__init__()

        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2,padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)


        self.layer1 = self.make_layer(block, 64, stride=1, num_blocks= num_blocks[0])
        self.layer2 = self.make_layer(block, 128, stride=2, num_blocks = num_blocks[1])
        self.layer3 = self.make_layer(block, 256, stride=2, num_blocks = num_blocks[2])
        self.layer4 = self.make_layer(block, 512, stride=2, num_blocks = num_blocks[3])



        self.fc = nn.Linear(in_features=1000, out_features= n_classes)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=n_classes)
            # nn.Linear(in_features=1000, out_features= n_classes)
        )


    def make_layer(self,block, out_planes, stride, num_blocks):
        layers=[]
        strides = [stride] + [1] * (num_blocks-1)

        for i in range(num_blocks):
            layers.append(block(self.in_planes,out_planes, stride=strides[i]))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        print('x : ', x.size())

        x = torch.relu(self.conv1(x))
        print('conv1 : ', x.size())


        x = torch.relu(self.maxpool(x))
        print('maxpool : ', x.size())

        x = self.layer1(x)
        print('layer1 : ', x.size())

        x = self.layer2(x)
        print('layer2 : ', x.size())

        x = self.layer3(x)
        print('layer3 : ', x.size())

        x = self.layer4(x)
        print('layer4 : ', x.size())

        x = self.avgpool(x)

        x = x.view(-1,512)
        logits = self.classifier(x)
        print('classifier : ', logits.size())

        probs = torch.softmax(logits, dim=1)
        return logits, probs

