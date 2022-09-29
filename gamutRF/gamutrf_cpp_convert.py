import torch

from gamutrf_model import *
from gamutrf_dataset import *


# Load model with pretrained weights
checkpoint_filepath = "gamutRF/model_weights/resnet18_pdx_and_leesburg_0.02_3.pt"
cpp_model_filepath = "gamutRF/cpp/traced_gamutrf_model.pt"

device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GamutRFModel(pretrained_weights=checkpoint_filepath, device=device)
model = model.to(device)
model.eval()
sample_secs = model.checkpoint["sample_secs"]
nfft = model.checkpoint["nfft"]

# Load GamutRF dataset
label_dirs= {
    'drone': ['data/gamutrf-birdseye-field-days/leesburg_field_day_2022_06_15/worker1/','data/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/worker1/gamutrf/'], 
    #'wifi_2_4': ['data/gamutrf-pdx/07_21_2022/wifi_2_4/'], 
    #'wifi_5': ['data/gamutrf-pdx/07_21_2022/wifi_5/']
}

try: 
    dataset = GamutRFDataset(label_dirs, sample_secs=sample_secs, nfft=nfft)
    #print("input shape = ",dataset[0][0].unsqueeze(0).shape)
    traced_script_module = torch.jit.trace(model, dataset[0][0].unsqueeze(0).to(device))
except:
    example_data = torch.ones((1,3,256,256))
    print(f"Error using provided data. Using data of shape (1, 3, 256, 256)")
    traced_script_module = torch.jit.trace(model, example_data.to(device))
traced_script_module.save(cpp_model_filepath)
print(f"\n\nSaved C++ enabled PyTorch model to {cpp_model_filepath}\n\n")