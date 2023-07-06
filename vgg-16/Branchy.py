import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

# Add Branches
class Branchy_VGG(nn.Module):
    def __init__(self, net):
        super(Branchy_VGG, self).__init__()
        self.inference_mode = False
        self.head = nn.Sequential()
        self.backbone = nn.Sequential()
        for i in range(17):
            self.head.add_module(f'{i}', net.features[i])
        for i in range(17,31):
            self.backbone.add_module(f'{i}', net.features[i])
        self.exit1 = nn.Sequential(
            nn.Linear(256 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 10),
            nn.Softmax(dim=1)
        )
        self.avgpool = net.avgpool
        self.classifier = net.classifier

        self.exit_threshold = torch.tensor([0.5], dtype=torch.float32)


    def set_inference_mode(self):
        self.inference_mode = True


    def set_training_mode(self):
        self.inference_mode = False


    def forward(self, x):
        if self.inference_mode is False:
            x = self.head(x)
            x1 = self.avgpool(x)
            x1= torch.flatten(x1, 1)
            x1 = self.exit1(x1)

            x = self.backbone(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x1, x
        else:
            flag = 0
            x = self.head(x)
            x1 = self.avgpool(x)
            x1 = torch.flatten(x1, 1)
            x1 = self.exit1(x1)

            entropy = -torch.sum(x1 * torch.log(x1))
            # print("entropy:",entropy)
            if entropy.to(device) < self.exit_threshold.to(device):
                flag = 1
                return x1, flag


            x = self.backbone(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x, flag
