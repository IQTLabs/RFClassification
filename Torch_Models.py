
import torch
import torch.nn as nn
import torchvision.models as models

### VGG features with Fully Connected Layer
class VGGFC(nn.Module):
    def __init__(self, num_classes):
        super(VGGFC,self).__init__()
        self.num_classes = num_classes
        self.vggfull = models.vgg16(pretrained=True)
        modules=list(self.vggfull.children())[:-1] # remove the fully connected layer & adaptive averaging
        self.vggfeats=nn.Sequential(*modules)
        
        for param in self.vggfeats.parameters():
            param.requires_grad_(False)
        
        self._fc = nn.Linear(25088, num_classes)
    def forward(self, x):
        if len(x.shape)==4:
            x = torch.moveaxis(x,-1, 1)
        else:
            x = torch.moveaxis(x, -1, 0)
        x = self.vggfeats(x)
        x = x.reshape(-1,25088)
        x = self._fc(x)
        
        return x

class ResNetFC(nn.Module):
    def __init__(self, num_classes):
        super(ResNetFC,self).__init__()
        self.num_classes = num_classes
        self.resnetfull = models.resnet50(pretrained=True)
        modules=list(self.resnetfull.children())[:-2] # remove the fully connected layer & adaptive averaging
        self.resnetfeats=nn.Sequential(*modules)
        
        for param in self.resnetfeats.parameters():
            param.requires_grad_(False)
        
        self._fc = nn.Linear(100352, num_classes)
    def forward(self, x):
        if len(x.shape)==4:
            x = torch.moveaxis(x,-1, 1)
        else:
            x = torch.moveaxis(x, -1, 0)
        x = self.resnetfeats(x)
        x = x.reshape(-1,100352)
        x = self._fc(x)
        
        return x