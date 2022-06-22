from torch import nn
import torch.nn.functional as F
import torch

# resolution size 224x224 -> 112x112 -> 56x56 -> 28x28 -> 14x14 -> 7x7
class VGGNet(nn.Module):
    def __init__(self,n_classes=1000):
        super(VGGNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)


        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)


        self.classifier = nn.Sequential(
            nn.Linear(in_features=7*7*512, out_features= 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=n_classes)
        )





    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        x = self.maxpool(x)

        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))

        x = self.maxpool(x)

        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv6(x))

        x = self.maxpool(x)

        x = torch.relu(self.conv7(x))
        x = torch.relu(self.conv8(x))
        x = torch.relu(self.conv8(x))

        x = self.maxpool(x)

        x = torch.relu(self.conv8(x))
        x = torch.relu(self.conv8(x))
        x = torch.relu(self.conv8(x))

        x = self.maxpool(x)

        x = x.view(-1, 7*7*512)
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=1)
        return logits, probs
