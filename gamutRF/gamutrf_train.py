import os
import numpy as np 
import zstandard
from tqdm import tqdm
import torch
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from timeit import default_timer as timer
import torchvision
from torchvision import datasets, models, transforms

from gamutrf.sample_reader import get_reader
from gamutrf.utils import parse_filename 

from gamutrf_dataset import * 



label_dirs= {
    'drone': ['data/gamutrf-birdseye-field-days/leesburg_field_day_2022_06_15/worker1/','data/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/worker1/gamutrf/'], 
    'wifi_2_4': ['data/gamutrf-pdx/07_21_2022/wifi_2_4/'], 
    'wifi_5': ['data/gamutrf-pdx/07_21_2022/wifi_5/']
}


sample_secs = 0.02
nfft = 512
batch_size = 8
num_workers = 19
num_epochs = 4
train_val_test_split = [0.75, 0.05, 0.20]
save_iter = 200
eval_iter = 10000
leesburg_split = True
experiment_name = 'leesburg_split'

dataset = GamutRFDataset(label_dirs, sample_secs=sample_secs, nfft=nfft)
print(dataset.idx_to_class)
#dataset.debug(590)

train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, (int(np.ceil(train_val_test_split[0]*len(dataset))), int(np.ceil(train_val_test_split[1]*len(dataset))), int(train_val_test_split[2]*len(dataset))))

if leesburg_split: 
    train_val_test_split = [0.77, 0.03, 0.20]
    all_except_leesburg = [i for (i, idx) in enumerate(dataset.idx) if not('leesburg' in idx[1] and 'field' in idx[1])] 
    dataset_sub = torch.utils.data.Subset(dataset, all_except_leesburg)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset_sub, (int(np.ceil(train_val_test_split[0]*len(dataset_sub))), int(np.ceil(train_val_test_split[1]*len(dataset_sub))), int(train_val_test_split[2]*len(dataset_sub))))
    just_leesburg = [i for (i, idx) in enumerate(dataset.idx) if 'leesburg' in idx[1]]
    leesburg_subset = torch.utils.data.Subset(dataset, just_leesburg)
    validation_dataset = torch.utils.data.ConcatDataset((validation_dataset,leesburg_subset))
    
print(f"{len(train_dataset)=}")
print(f"{len(validation_dataset)=}")
print(f"{len(test_dataset)=}")
print(f"total len = {len(train_dataset)+len(validation_dataset)+len(test_dataset)}")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, num_workers=num_workers)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
# x,y = next(iter(dataloader))
# for x,y in zip(x,y): 
#     plt.imshow(np.moveaxis(x.cpu().numpy(), 0, -1), aspect='auto', origin='lower', cmap=plt.get_cmap('jet'))
#     plt.colorbar()
#     plt.title(f"{dataset.idx_to_class[y.item()]}")
#     plt.show()

model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.class_to_idx))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters(): 
    param.requires_grad = True
    
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')

    model.train()  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0
    start = timer()
    for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
        print(f"epoch {epoch}/{num_epochs}, iter {i}/{len(train_dataloader)}")
        #print(f"load training data = {timer()-start} seconds")
    
        start = timer() 
        inputs = inputs.to(device)
        labels = labels.to(device)
        #print(f".to(device) = {timer()-start} seconds")
        optimizer.zero_grad()

        start = timer() 
        outputs = model(inputs)
        #print(f"inference = {timer()-start} seconds")
        start = timer() 
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
            checkpoint_path = f"resnet18_{experiment_name}_{str(0.02)}_{epoch}_current.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'dataset_idx_to_class': dataset.idx_to_class,
            }, checkpoint_path)

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
            disp.figure_.savefig(f"confusion_matrix_{experiment_name}_{epoch}_{i}.png")
            model.train()
    epoch_loss = running_loss / (len(train_dataloader)*batch_size)
    epoch_acc = running_corrects.double() / (len(train_dataloader)*batch_size)

    print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    checkpoint_path = f"resnet18_{experiment_name}_{str(0.02)}_{epoch}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'dataset_idx_to_class': dataset.idx_to_class,
    }, checkpoint_path)
    
# Visualize predictions 
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
# model.eval()
# x,y = next(iter(dataloader))
# print(dataset.idx_to_class)
# for x,y in zip(x,y): 
    
#     test_x = x.unsqueeze(0).to(device)
#     test_y = y.unsqueeze(0).to(device)
#     out = model(test_x)

#     _, preds = torch.max(out, 1)
#     correct = preds == test_y.data
#     print(f"out={out}")
#     print(f"label={test_y.item()}, prediction={preds.item()}")
#     print(f"correct={correct.item()}")
    
#     plt.imshow(np.moveaxis(x.cpu().numpy(), 0, -1), aspect='auto', origin='lower', cmap=plt.get_cmap('jet'))
#     plt.colorbar()
#     plt.title(f"{dataset.idx_to_class[y.item()]}")
#     plt.show()
