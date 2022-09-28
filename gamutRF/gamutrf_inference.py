from sklearn.metrics import ConfusionMatrixDisplay

from gamutrf_model import * 
from gamutrf_dataset import * 


# Load model with pretrained weights
weights_filepath = "gamutRF/model_weights/resnet18_0.02_3.pt"
model = GamutRFModel(3, weights_filepath)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Load GamutRF dataset
label_dirs= {
    'drone': ['data/gamutrf-birdseye-field-days/leesburg_field_day_2022_06_15/worker1/','data/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/worker1/gamutrf/'], 
    'wifi_2_4': ['data/gamutrf-pdx/07_21_2022/wifi_2_4/'], 
    'wifi_5': ['data/gamutrf-pdx/07_21_2022/wifi_5/']
}
sample_secs = 0.02
nfft = 512
train_val_test_split = [0.75, 0.05, 0.20]
batch_size = 8
num_workers = 0

dataset = GamutRFDataset(label_dirs, sample_secs=sample_secs, nfft=nfft)
print(f"\n\n\nDataset class to idx mapping: {dataset.idx_to_class}\n\n\n")
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, (int(np.ceil(train_val_test_split[0]*len(dataset))), int(np.ceil(train_val_test_split[1]*len(dataset))), int(train_val_test_split[2]*len(dataset))))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

predictions = []
labels = []
with torch.no_grad():
    for j,(data,label) in enumerate(tqdm(test_dataloader)): 
        #print(f"testing {j}/{len(test_dataloader)}")

        data = data.to(device)
        label = label.to(device)
        out = model(data)

        _, prediction = torch.max(out, 1)
        predictions.extend(prediction.detach().cpu().numpy())
        labels.extend(label.detach().cpu().numpy())

        #correct = preds == label.data

disp = ConfusionMatrixDisplay.from_predictions(labels, predictions, display_labels=list(dataset.class_to_idx.keys()), normalize='true')
disp.figure_.savefig(f"confusion_matrix_test_set.png")
