import torch
import torchvision
from torchvision import models


class GamutRFModel(torch.nn.Module):
    def __init__(self, n_classes, pretrained_weights=None): 
            super().__init__()
            if pretrained_weights is None: 
                self.model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, n_classes)
            else: 
                self.model = models.resnet18()
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, n_classes)
                self.model.load_state_dict(torch.load(pretrained_weights))

    def forward(self, x):
        return self.model(x)