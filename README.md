# RFClassification
Project aimed to use RF signals to detect and classify devices

## General Description
This project is aimed to use RF signals to detect and classify devices. 
The current repository explores using RF signals from drone communications to classify the presence of and type of drones.
The project currently uses training data from [DroneRF](https://www.sciencedirect.com/science/article/pii/S2352340919306675?ref=cra_js_challenge&fr=RR-1) and [DroneDetect](https://ieee-dataport.org/open-access/dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine#files) Datasets.
Additionally, the project aims to test models using data from [GamutRF](https://github.com/IQTLabs/gamutRF) hardware.
The eventual goal is to detect and classify drones in real-time using GamutRF.

The current repository contains methods derived from published RF classication papers.

## Setup
### Data
Download [*DroneRF*](https://www.sciencedirect.com/science/article/pii/S2352340919306675?ref=cra_js_challenge&fr=RR-1) and [*DroneDetect*](https://ieee-dataport.org/open-access/dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine#files) datasets to local directory.
If available, also locate data collected by GamutRF workers.
For all dataset, update the path the the datasets in file_path.py


### Libraries
* *scikit-learn*: machine learning in Python
* *PyTorch*: machine learning & deep learning
* *scipy* and *matplotlib*: for computing rf-based features
* other packages: *tqdm*

## Files & Descriptions
The files in this repository can be grouped into the following categories. Most files are developed in iPython notebooks and transfered to Python files (some files serve the same purpose).

- **Feature Generation**
  - *Generate DroneDetect Features.ipynb* and *run_dronedetect_feat.py* 
    - generate PSD, spectrogram features from DroneDetect I/Q data
  - *Generate DroneRF Features.ipynb* and *run_dronerf_feat.py*
    - Generate PSD, spectrogram features from DroneRF real data
  - *Generate GamutRF Features.ipynb*
    - Generate PSD, spectrogram from field day data collected by GamutRF

- **Models** (preliminary model exploration)
  - *ML Approaches.ipynb*
    - PSD features + SVM
  - *DL Approaches.ipynb* and *run_dl.py*
    - Transfer learning models with ResNet50 and VGG16
  - *RFUAV-Net.ipynb* and *run_rfuav.py*
    - 1DConv model from RFUAVNet paper
  - *TransferLearning_LRkNN.ipynb*
    - Transfer learning model with logistic ression and kNN classifiers (with dimensionality reduction exploration)
  - *Kilic Paper Implementation.ipynb
    - Kilic2021 paper implementation (containing both feature generation and model implementation)
  - **Note model classes are in : models.py

- **Helper funcutions**
  - *loading_functions.py*
    - contains dataloaders that loads PSD, spectrogram features and raw data from all datasets
  - *helper_functions.py*
    - data plotting & other misc tasks
  - *latency_functions.py*
    - measure inference time
  - *nn_functions.py*
    - run k-fold CV for PyTorch-based models
  - *feat_gen_functions.py*
    - helpers for feature generation notebooks
  - *gamutrf_feature_functions.py*
    - feature generation helper functions for GamutRF data
 
 - **test_code_on_pi/** contains scripts to ran on RPI 4 for real time prediction feasibility test 
 - **tests/** directory
    - test scripts for development
 - **Semi-supervised Methods/**
    - methods exploring unsupervised classification
 - **gamutrf/**
    - dataloader and models for gamutrf data
  - **archive/** old development notebooks
  - **images/** sample PSD and Spectrogram images
    

## How to run
### 1. Generate Features
Run *Generate DroneDetect Features.ipynb* (*run_dronedetect_feat.py*) or, *Generate DroneRF Features.ipynb* (*run_dronerf_feat.py*) or  or *Generate GamutRF Features.ipynb* to compute and save features to directory.
Choose which features (Power Spectral Density (PSD) or Spectrogram), feature specifications (N_FFT, N_Overlap, sample duration), and feature format (array or images).
Update which directory to save these features in in *file_paths.py*

In array format, PSD features are saved in 1d float arrays and Spectrogram features are saved in 2D arrays with size determined by N_FFT and N_Overlap.

In image format, plots of PSD and Spectrograms are saved in .jpg format without axis labels. Examples plots from DroneDetect dataset shown here:

* PSD from DJI Air 2S when switched on with wifi and bluetooth interference
<img src="https://github.com/IQTLabs/RFClassification/blob/main/images/AIR_ON_11_00_60.jpg" alt="drawing" width="250"/>

* Spectrogram from DJI Air 2S when switched on with wifi interference
<img src="https://github.com/IQTLabs/RFClassification/blob/main/images/AIR_ON_10_04_87.jpg" alt="drawing" width="250"/>

### 2. Apply Models
Run the model notebooks and scripts, which loads features (& normalize if applicable), preform train-test split, train and test models.

*ML Approaches.ipynb* uses PSD feature + SVM model derived from [Drone classification using RF signal based spectral features](https://www.sciencedirect.com/science/article/pii/S2215098621001403).

*DL Approaches.ipynb* and *run_dl.py* includes transfer learning models using features from inputting spectrogram and PSD images to trained VGG/ResNet models and then applying a final fully connected layer. These models are inspired from [The Effect of Real-World Interference on CNN Feature Extraction and Machine Learning Classification of Unmanned Aerial Systems](https://www.proquest.com/openview/ff99105f660c7fe97afae45f8a384c04/1?pq-origsite=gscholar&cbl=2032442#:~:text=In%20the%20presence%20of%20interference,mode%20classification%20(21%20classes)) and [Unmanned Aerial Vehicle Operating Mode Classification Using Deep Residual Learning Feature Extraction](https://www.mdpi.com/2226-4310/8/3/79).

*RFUAV-Net.ipynb* and *run_rfuav.py* includes an implementation of a 1D convolution network named RFUAV Net from [RF-UAVNet: High-Performance Convolutional Network for RF-Based Drone Surveillance Systems](https://ieeexplore.ieee.org/document/9768809). Raw DroneRF data is used for this model.

## Result Highlights
**Inference time measured on workstation with 128Gb RAM, Intel Core i9-9820X CPU and 1 Titan RTX

### Binary Drone Detection Results

| Dataset | Sample Length | Model                           | Accuracy | F1 Score | Inference Time** |
|---------|---------------|---------------------------------|----------|----------|----------------|
| DroneRF (High Freq) | 20 ms          | PSD(NFFT=1024) + SVM            | 0.983       | 0.982        | 0.286ms         |
| DroneRF (High & Low Freq) | 0.025ms       | Raw data + 1D Conv (RF-UAV Net) | 0.998    | 0.998    | 1.078ms         |


### Multiclass Drone Type Classification Results
| Dataset | Sample Length | Model                           | Accuracy | F1 Score | Inference Time**|
|---------|---------------|---------------------------------|----------|----------|----------------|
| DroneDetect | 20ms          | PSD(NFFT=1024) + SVM | 0.854        | 0.851        |9.96ms         |
| DroneDetect | 20ms       | SPEC(NFFT=1024)+VGG16+Linear | 0.816       | 0.818         |  5.727ms        |                |
| DronDetect        |  20ms              |      PSD(NFFT=1024)+VGG16+Linear                          |  0.825        |   0.825       | 5.721ms               |

## Comparison of Model Parameters
For PSD+SVM model with DroneDetect data for drone type classification, we compared the performance of model with different sample lengths and FFT lengths.
The preliminary results show, FFT lenghts does not significantly vary performance, whereas longer time samples result in better performance.

Model Performance (accuracy) for different NFFT and sample lengths

|       | NFFT = 256 | NFFT = 512 | NFFT = 1024 | 
|-------|------------|------------|-------------|
| 20 ms | 0.841       | 0.840       | 0.855        |


|       | 10ms | 20ms | 50ms | 
|-------|------------|------------|-------------|
| NFFT=1024 | 0.769       | 0.836       | 0.894        |

