import torch
import torch.nn as nn
import torchvision.models as models


class KQnet(nn.Module):
    def __init__(self):
        super(KQnet, self).__init__()
        model = models.resnet18(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x




