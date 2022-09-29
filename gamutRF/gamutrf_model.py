import torch
import torchvision
from torchvision import models


class GamutRFModel(torch.nn.Module):
    def __init__(self, 
        experiment_name=None, 
        sample_secs=None, 
        nfft=None, 
        label_dirs=None, 
        dataset_idx_to_class=None, 
        pretrained_weights=None, 
        device=None
    ): 
        super().__init__()
        if pretrained_weights is None: 
            assert experiment_name is not None
            assert sample_secs is not None
            assert nfft is not None
            assert label_dirs is not None
            assert dataset_idx_to_class is not None

            self.model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            n_classes = len(dataset_idx_to_class)
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, n_classes)
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters(): 
                param.requires_grad = True
           
            self.checkpoint = ({
                "experiment_name": experiment_name, 
                "sample_secs": sample_secs, 
                "nfft": nfft,
                "label_dirs": label_dirs,
                "dataset_idx_to_class": dataset_idx_to_class,
            })

        else: 
            if device is None: 
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.checkpoint = torch.load(pretrained_weights, map_location=device)

            model_weights = self.checkpoint["model_state_dict"]
            n_classes = len(self.checkpoint["dataset_idx_to_class"])

            self.model = models.resnet18()
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, n_classes)
            self.model.load_state_dict(model_weights)
            self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)

    def save_checkpoint(self, checkpoint_path): 
        self.checkpoint["model_state_dict"] = self.model.state_dict()
        torch.save(self.checkpoint, checkpoint_path)