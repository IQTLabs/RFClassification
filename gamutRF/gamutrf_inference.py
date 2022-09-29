from sklearn.metrics import ConfusionMatrixDisplay

from gamutrf_model import * 
from gamutrf_dataset import * 


# Load model with pretrained weights
#weights_filepath = "gamutRF/model_weights/resnet18_pdx_and_leesburg_0.02_3.pt"
weights_filepath = "gamutRF/model_weights/resnet18_leesburg_split_0.02_1_current.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GamutRFModel(pretrained_weights=weights_filepath, device=device)
model.eval()
sample_secs = model.checkpoint["sample_secs"]
nfft = model.checkpoint["nfft"]

# Set the directory for IQ samples to be used in inference
label_dirs= {
    'drone': ['data/gamutrf-birdseye-field-days/leesburg_field_day_2022_06_15/worker1/','data/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/worker1/gamutrf/'], 
    'wifi_2_4': ['data/gamutrf-pdx/07_21_2022/wifi_2_4/'], 
    'wifi_5': ['data/gamutrf-pdx/07_21_2022/wifi_5/']
}

# Create inference dataset
train_val_test_split = [0.75, 0.2499, 0.05]
batch_size = 1
num_workers = 0
dataset = GamutRFDataset(label_dirs, sample_secs=sample_secs, nfft=nfft, idx_to_class=model.checkpoint["dataset_idx_to_class"])
n_train = int(np.floor(train_val_test_split[0]*len(dataset)))
n_validation = int(np.floor(train_val_test_split[1]*len(dataset)))
n_test = len(dataset) - n_train - n_validation
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, (n_train, n_validation, n_test))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

predictions = []
labels = []

# Run inference 
print(f"\n\n\nLoaded saved model {weights_filepath} with the following training settings:")
print(f"Inference sample seconds = {sample_secs}")
print(f"NFFT = {nfft}")
print(f"Dataset class to idx mapping: {dataset.idx_to_class}")
print(f"Training directories = {model.checkpoint['label_dirs']}\n\n\n")
with torch.no_grad():
    for j,(data,label) in enumerate(tqdm(test_dataloader)): 
        data = data.to(device)
        label = label.to(device)

        out = model(data)
        _, prediction = torch.max(out, 1)
        
        predictions.extend(prediction.detach().cpu().numpy())
        labels.extend(label.detach().cpu().numpy())

        #correct = preds == label.data

# Display results as confusion matrix 
disp = ConfusionMatrixDisplay.from_predictions(labels, predictions, display_labels=[model.checkpoint["dataset_idx_to_class"][y] for y in np.unique(predictions+labels)], normalize='true')
disp.figure_.savefig(f"confusion_matrix_inference_{model.checkpoint['experiment_name']}.png")
