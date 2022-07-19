# RFClassification
Project aimed to use RF signals to detect and classify devices

## General Description
This project is aimed to use RF signals to detect and classify devices. 
The current repository explores using RF signals from drone communications to and classify the presense and the types of drones.
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
* *scipy* and *matplotlib*: for computing rf-based features

## Files & Descriptions
The files in this repository can be grouped into the following categories.
<img src="https://github.com/IQTLabs/RFClassification/blob/main/images/repo_structure_image_0719.jpg" alt="drawing" width="800"/>

## How to run
### 1. Generate Features
Run *Generate DroneDetect Features.ipynb*, *Generate DroneRF Features.ipynb* or *Generate GamutRF Features.ipynb* to compute and save features to directory.
Choose which features (Power Spectral Density (PSD) or Spectrogram), feature specifications (N_FFT, N_Overlap, sample duration), and feature format (array or images).
Select/Edit the directory to save features in.

In array format, PSD features are saved in 1d float arrays and Spectrogram features are saved in 2D arrays with size determined by N_FFT and N_Overlap.

In image format, plots of PSD and Spectrograms are saved in .jpg format without axis labels. Examples plots from DroneDetect dataset shown here:

* PSD from DJI Air 2S when switched on with wifi and bluetooth interference
<img src="https://github.com/IQTLabs/RFClassification/blob/main/images/AIR_ON_11_00_60.jpg" alt="drawing" width="250"/>

* Spectrogram from DJI Air 2S when switched on with wifi interference
<img src="https://github.com/IQTLabs/RFClassification/blob/main/images/AIR_ON_10_04_87.jpg" alt="drawing" width="250"/>

### 2. Apply Models
Run the model notebooks. *ML Approaches.ipynb* include PSD feature + SVM model derived from [Drone classification using RF signal based spectral features](https://www.sciencedirect.com/science/article/pii/S2215098621001403).

*DL Approaches.ipynb* include transfer learning models using features generated from spectrogram and PSD through trained VGG/ResNet and LR model. These models are from [The Effect of Real-World Interference on CNN Feature Extraction and Machine Learning Classification of Unmanned Aerial Systems](https://www.proquest.com/openview/ff99105f660c7fe97afae45f8a384c04/1?pq-origsite=gscholar&cbl=2032442#:~:text=In%20the%20presence%20of%20interference,mode%20classification%20(21%20classes)) and [Unmanned Aerial Vehicle Operating Mode Classification Using Deep Residual Learning Feature Extraction](https://www.mdpi.com/2226-4310/8/3/79).

*RFUAV-Net.ipynb* incldue an implementation of a 1D convolution model named RFUAV Net from [RF-UAVNet: High-Performance Convolutional Network for RF-Based Drone Surveillance Systems](https://ieeexplore.ieee.org/document/9768809)

## Result Highlights
**** preliminary, to be completed

### Binary Drone Detection Results
| Dataset | Sample Length | Model                           | Accuracy | F1 Score | Inference Time |
|---------|---------------|---------------------------------|----------|----------|----------------|
| DroneRF | 50ms          | PSD(NFFT=1024) + SVM            | 1        | 1        | 0.13ms         |
| DroneRF | 0.025ms       | Raw data + 1D Conv (RF-UAV Net) | 0.996    | 0.994    | 1.48ms         |
|         |               |                                 |          |          |                |


### Multiclass Drone Type Classification Results
| Dataset | Sample Length | Model                           | Accuracy | F1 Score | Inference Time |
|---------|---------------|---------------------------------|----------|----------|----------------|
| DroneDetect | 200ms          | PSD(NFFT=512) + SVM | 0.94        | 0.94        | 0.66ms         |
| DroneDetect | 20ms       | PSD(NFFT=1024)+VGG16+LR                     |          |          |                |
|         |               |                                 |          |          |                |

## Comparison of Model Parameters
*** to be completed
