import torch
import torchvision
from torchvision import models


class GamutRFModel(torch.nn.Module):
    def __init__(self, n_classes, pretrained_weights=None, device=None): 
            super().__init__()
            if pretrained_weights is None: 
                self.model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, n_classes)
            else: 
                self.model = models.resnet18()
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, n_classes)
                if device is None: 
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.model.load_state_dict(torch.load(pretrained_weights, map_location=device))

    def forward(self, x):
        return self.model(x)