import torch
import torchvision
from torchvision import models


class GamutRFModel(torch.nn.Module):
    def __init__(self, n_classes=None, pretrained_weights=None, device=None): 
        super().__init__()
        if pretrained_weights is None: 
            assert n_classes is not None
            self.model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, n_classes)
        else: 
            if device is None: 
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(pretrained_weights, map_location=device)
            model_weights = checkpoint["model_state_dict"]
            self.idx_to_class = checkpoint["dataset_idx_to_class"]
            n_classes = len(self.idx_to_class)

            self.model = models.resnet18()
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, n_classes)
            self.model.load_state_dict(model_weights)

    def forward(self, x):
        return self.model(x)