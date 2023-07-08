import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

# Add Branches
class Branchy_ResNet(nn.Module):
    def __init__(self, net):
        super(Branchy_ResNet, self).__init__()
        self.inference_mode = False
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avgpool = net.avgpool
        self.fc = nn.Sequential(
            nn.Linear(2048, 100),
            nn.Softmax(dim=1)
        )
        ## Exit Branch 1
        self.exit1 = nn.Sequential(
            nn.Linear(64, 100),
            nn.Softmax(dim=1)
        )
        ## Exit Branch 2
        self.exit2 = nn.Sequential(
            nn.Linear(512, 100),
            nn.Softmax(dim=1)
        )

        self.exit1_threshold = torch.tensor([4.575], dtype=torch.float32)
        self.exit2_threshold = torch.tensor([4.601], dtype=torch.float32)


    def set_inference_mode(self):
        self.inference_mode = True


    def set_training_mode(self):
        self.inference_mode = False


    def forward(self, x):
        if self.inference_mode is False:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x1 = self.avgpool(x)
            x1 = torch.flatten(x1, 1)
            x1 = self.exit1(x1)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)

            x2 = self.avgpool(x)
            x2 = torch.flatten(x2, 1)
            x2 = self.exit2(x2)

            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x1, x2, x
        else:
            flag = 0
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x1 = self.avgpool(x)
            x1 = torch.flatten(x1, 1)
            x1 = self.exit1(x1)
            entropy = -torch.sum(x1 * torch.log(x1))
            # print("entropy:", entropy)
            if entropy.to(device) < self.exit1_threshold.to(device):
                flag = 1
                # print("Early Exit!")
                return x1, flag

            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)

            x2 = self.avgpool(x)
            x2 = torch.flatten(x2, 1)
            x2 = self.exit2(x2)

            entropy = -torch.sum(x2 * torch.log(x2))
            # print("entropy:", entropy)
            if entropy.to(device) < self.exit2_threshold.to(device):
                flag = 1
                # print("Early Exit!")
                return x2, flag

            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            # entropy = -torch.sum(x * torch.log(x))
            # print("entropy:", entropy)

            # print("Exit!")
            return x, flag
