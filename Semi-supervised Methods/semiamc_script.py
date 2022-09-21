## SemiAMC model by Liu2022 from notebook
## Transfer to script form to run overnight

import sys; sys.path.insert(0, '..') # add parent folder path where lib folder is

# import custom functions
from helper_functions import *
from latency_helpers import *
from loading_functions import *
from gamutRF.gamutrf_dataset import *
from SemiSupervised_Models import *

# import the torch packages
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision.models as models

import random
import time


label_dirs= {
    'drone': ['/home/ltindall/data/gamutrf-birdseye-field-days/leesburg_field_day_2022_06_15/worker1/',
              '/home/ltindall/data/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/worker1/gamutrf/'], 
    'wifi_2_4': ['/home/ltindall/data/gamutrf-pdx/07_21_2022/wifi_2_4/'], 
    'wifi_5': ['/home/ltindall/data/gamutrf-pdx/07_21_2022/wifi_5/']
}


sample_secs = 0.02
nfft = 1024
batch_size = 8
num_workers = 19
num_epochs =1
train_val_test_split = [0.75, 0.05, 0.2] #[0.0005, 0.0005, 0.999]  #[0.75, 0.05, 0.20] ##[0.75, 0.05, 0.20]
save_iter = 200
eval_iter = 10000
feat = 'iq'

dataset = GamutRFDataset(label_dirs, sample_secs=sample_secs, nfft=nfft, feat=feat)
print(dataset.idx_to_class)

print('Number of samples:', len(dataset), "Shape of each shape", dataset.__getitem__(0)[0].shape)

### How many samples of each label?
for lb in dataset.unique_labels:
    print(lb, ' has', sum(didx[0]==lb for didx in dataset.idx))

# ### Set propotion of known and unknown data
labeled_splits = {}
labeled_splits['drone'] = [0.5, 0.5] # [number_labelled, number_unlabelled]
labeled_splits['wifi_2_4'] = [1, 0]
labeled_splits['wifi_5'] = [1, 0]


### train_val_test_split
# split whole sample numbers
train_val_test_split_wsamp = [None] * len(train_val_test_split)
for i in range(len(train_val_test_split)-1):
    train_val_test_split_wsamp[i] = int(np.ceil(train_val_test_split[i]*len(dataset)))
train_val_test_split_wsamp[-1] = len(dataset) - sum(train_val_test_split_wsamp[:-1])

train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, 
                                                                                train_val_test_split_wsamp)

# split into known and unknonwn labels
# known_dataset, unknown_dataset = split_knownunknown_by_label(train_dataset, labeled_splits)
# ^ this is taking a long time

uk_split_samps = [int(train_val_test_split_wsamp[0]*ls) for ls in labeled_splits['drone']]
uk_split_samps[-1] = len(train_dataset) - sum(uk_split_samps[:-1])
print('Known vs. Unknonwn Split:', uk_split_samps)

known_data, unknown_data = torch.utils.data.random_split(train_dataset,
                                                         uk_split_samps)

# get data loaders
batch_size = 8
num_workers = 10
unknown_dataloader = torch.utils.data.DataLoader(unknown_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
known_dataloader = torch.utils.data.DataLoader(known_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

##### TRAIN ENCODER #####
num_classes = 3
model = ContrastiveEncoder(num_classes, to_augment=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for name, p in model.named_parameters():
    if "proj" in name: # freeze the project layers
        print('Not training for', name)
        p.requires_grad = False
    else:
        p.requires_grad = True

# Train the encoder
print('------- TRAINING ENCODER -------')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')

    model.train()  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0
    start = timer()
    for i, (inputs, labels) in enumerate(tqdm(unknown_dataloader)):
        print(f"epoch {epoch}/{num_epochs-1}, iter {i}/{len(unknown_dataloader)}")
        print(f"load training data = {timer()-start} seconds")
    
        start = timer() 
        inputs = inputs.to(device)
        labels = labels.to(device)
        print(f".to(device) = {timer()-start} seconds")
        optimizer.zero_grad()

        start = timer() 
        # 2 augmentations
        output_1 = model.forward(inputs)
        output_2 = model.forward(inputs)
        
        print(f"inference = {timer()-start} seconds")
        start = timer() 
        loss = contrastive_loss(output_1, output_2)

        #print(f"loss = {timer() - start} seconds")
#         _, preds = torch.max(outputs, 1)
#         correct = torch.sum(preds == labels.data)
        print(f"loss={loss.item()}")

        start = timer() 
        loss.backward()
        optimizer.step()
        #print(f"backward and step = {timer()-start} seconds")

#### Train classifier ####
print('------- TRAINING CLASSIFIER -------')
# freeze the encoder layer (flip the layers requiring grad)
for name, param in model.named_parameters():
    param.requires_grad = not param.requires_grad
    print(param.requires_grad)

## Run again with actual labels
model.to_augment = False
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')

    model.train()  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0
    start = timer()
    for i, (inputs, labels) in enumerate(tqdm(known_dataloader)):
        print(f"epoch {epoch}/{num_epochs}, iter {i}/{len(known_dataloader)}")
        #print(f"load training data = {timer()-start} seconds")
    
        start = timer() 
        inputs = inputs.to(device)
        labels = labels.to(device)
        #print(f".to(device) = {timer()-start} seconds")
        optimizer.zero_grad()

        start = timer() 
        # 2 augmentations
        outputs = model.forward(inputs)
#         output_2 = model.forward(inputs)
        
        #print(f"inference = {timer()-start} seconds")
        start = timer() 
#         loss = contrastive_loss(output_1, output_2)
        loss = criterion(outputs, labels)
        #print(f"loss = {timer() - start} seconds")
        _, preds = torch.max(outputs, 1)
        correct = torch.sum(preds == labels.data)
        print(f"loss={loss.item()}, accuracy={correct/len(preds)}")

        start = timer() 
        loss.backward()
        optimizer.step()
        
        #print(f"backward and step = {timer()-start} seconds")

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        start = timer() 
        
        if (i+1)%save_iter == 0: 
            model_path = f"resnet18_{str(0.02)}_{epoch}_current.pt"
            torch.save(model.state_dict(), model_path)
        if (i+1)%eval_iter == 0: 
            model.eval()
            predictions = []
            labels = []
            with torch.no_grad():
                for j,(data,label) in enumerate(validation_dataloader): 
                    print(f"validating {j}/{len(validation_dataloader)}")

                    data = data.to(device)
                    label = label.to(device)
                    out = model(data)

                    _, preds = torch.max(out, 1)
                    predictions.append(preds.item())
                    labels.append(label.item())
                    correct = preds == label.data
            disp = ConfusionMatrixDisplay.from_predictions(labels, predictions, display_labels=list(dataset.class_to_idx.keys()), normalize='true')
            disp.figure_.savefig(f"figures/rawiq_confusion_matrix_{epoch}_{i}.png")
            model.train()
    epoch_loss = running_loss / (len(known_dataloader)*batch_size)
    epoch_acc = running_corrects.double() / (len(known_dataloader)*batch_size)

    print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
#     model_path = f"resnet18_{str(0.02)}_{epoch}.pt"
#     torch.save(model.state_dict(), model_path)