from torch import nn
import torch.nn.functional as F
import torch

# resolution size 32x32 -> 28x28 -> 14x14 -> 10x10 -> 5x5
class LeNet_5(nn.Module):
    def __init__(self,n_classes):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(1,6, kernel_size=5, stride=1)
        self.avg = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features= 84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes)
        )





    def forward(self, x):

        x = torch.tanh(self.conv1(x))

        x = torch.tanh(self.avg(x))


        x = torch.tanh(self.conv2(x))


        x = torch.tanh(self.avg(x))


        x = torch.tanh(self.conv3(x))


        x = x.view(-1,120)
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=1)
        return logits, probs

