# RFClassification
Project aimed to use RF signals to detect and classify devices

## General Description
This project is aimed to use RF signals to detect and classify devices. 
The current repository explores using RF signals from drone communications to and classify presense and the type of drones.
The project currently uses training data from [DroneRF](https://www.sciencedirect.com/science/article/pii/S2352340919306675?ref=cra_js_challenge&fr=RR-1) and [DroneDetect](https://ieee-dataport.org/open-access/dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine#files) Datasets.
Additionally, the project aims to test models using data from [GamutRF](https://github.com/IQTLabs/gamutRF) hardware.
The eventual goal is to detect and classify drones in real-time using GamutRF.

The current repository mainly uses methods derived from published RF classication papers.

## Setup
### Data
Download [*DroneRF*](https://www.sciencedirect.com/science/article/pii/S2352340919306675?ref=cra_js_challenge&fr=RR-1) and [*DroneDetect*](https://ieee-dataport.org/open-access/dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine#files) datasets to local directory.
If available, also locate data collected by GamutRF workers.

### Libraries
* *scikit-learn*: machine learning in Python
* *PyTorch*: machine learning & deep learning
* *scipy*: for computing rf-based features

## Files & Descriptions
*** to insert image ***

## How to run
### 1. Generate Features
Run *Generate DroneDetect Features.ipynb*, *Generate DroneRF Features.ipynb* or *Generate GamutRF Features.ipynb* to compute and save features to directory.
Choose which features (Power Spectral Density (PSD) or Spectrogram), feature specifications (N_FFT, N_Overlap, sample duration), and feature format (array or images).
Select/Edit the directory to save features in.

In array format, PSD features are saved in 1d float arrays and Spectrogram features are saved in 2D arrays with size determined by N_FFT and N_Overlap.
In image format, plots of PSD and Spectrograms are saved in .jpg format without axis labels. Examples plots from DroneDetect dataset shown here:

*** Insert images ***

### 2. Apply Models
Run the model notebooks. *ML Approaches.ipynb* include PSD feature + SVM model derived from [Drone classification using RF signal based spectral features](https://www.sciencedirect.com/science/article/pii/S2215098621001403).
*DL Approaches.ipynb* include transfer learning models using features generated from spectrogram and PSD through trained VGG/ResNet and LR model. These models are from The Effect of Real-World Interference on CNN Feature Extraction and Machine Learning Classification of Unmanned Aerial Systems and Unmanned Aerial Vehicle Operating Mode Classification Using Deep Residual Learning Feature Extraction.
*RFUAV-Net.ipynb* incldue an implementation of a 1D convolution model named RFUAV Net from [RF-UAVNet: High-Performance Convolutional Network for RF-Based Drone Surveillance Systems](https://ieeexplore.ieee.org/document/9768809)

## Preliminary Results
**** to be completed
### Binary Drone Detection Results
| Dataset | Model                | Accuracy | F1 Score | Inference Time |
|---------|----------------------|----------|----------|----------------|
| DroneRF | PSD(NFFT=1024) + SVM | 1        | 1        | 0.13ms         |
|         |                      |          |          |                |
|         |                      |          |          |                |

### Multiclass Drone Type Classification Results
| Dataset | Model                | Accuracy | F1 Score | Inference Time |
|---------|----------------------|----------|----------|----------------|
| DroneDetect | PSD(NFFT=512) + SVM | 0.94        | 0.94        | 0.66ms         |
|         |                      |          |          |                |
|         |                      |          |          |                |
