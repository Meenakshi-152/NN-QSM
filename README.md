# NN-QSM: Physics-Guided Neumann Network for Quantitative Susceptibility Mapping

## Overview

Quantitative Susceptibility Mapping (QSM) is an MRI-based technique used to estimate the magnetic susceptibility distribution of biological tissues from measured local magnetic field maps. QSM reconstruction is an ill-posed inverse problem, and obtaining accurate susceptibility maps remains challenging because of noise, artifacts, and the ill-conditioned nature of the dipole inversion problem.

This repository contains the implementation of **NN-QSM**, a physics-guided Neumann Network for QSM reconstruction. The proposed approach incorporates the dipole forward model into a truncated Neumann-series formulation and uses a convolutional neural network (CNN) to learn the regularization component.

By combining the physical model with learned regularization, NN-QSM aims to produce accurate susceptibility maps while avoiding the iterative optimization procedure required by some model-based reconstruction methods during inference.

## Method

NN-QSM combines the following components:

- **Physics-guided reconstruction:** Incorporates the dipole forward model into the reconstruction network using Fourier-domain operations.
- **Truncated Neumann-series formulation:** Approximates the inverse problem using a finite number of Neumann blocks.
- **Learned regularization:** Uses a CNN-based regularizer to capture spatial features relevant to susceptibility reconstruction.
- **Quantitative evaluation:** Evaluates reconstruction quality using metrics such as SSIM, PSNR, RMSE, and HFEN.

The truncation depth controls the number of Neumann blocks used in the reconstruction. Its selection involves balancing reconstruction quality and computational cost.

## Repository Structure

The repository will contain the following components as they are added:

```text
The repository is organized as follows:

NN-QSM/
│
├── checkpoints/
│   ├── full_data/
│   │   ├── fold1/
│   │   ├── fold2/
│   │   ├── fold3/
│   │   └── fold4/
│   │
│   └── lim_data/
│
├── figures/
│
├── qsm_modules/
│   ├── qsm_data_loader/
│   ├── qsm_dw_models/
│   └── qsm_loss_modules/
│
├── scripts/
│   ├── full_data/
│   │   ├── train_amp.py
│   │   └── Neumann_QSM_test_new.py
│   │
│   └── lim_data/
│       ├── train_amp_lim.py
│       └── Neumann_QSM_test_new_lim.py
│
├── .gitignore
├── requirements.txt
└── README.md
```
# Directory Description

checkpoints/ – Stores trained model checkpoints and experimental results. The full_data/ directory contains results from four-fold experiments, while lim_data/ contains results from limited-data experiments.

figures/ – Contains figures and visualizations related to the network architecture and experimental results.

qsm_modules/ – Contains the core implementation of NN-QSM, including data loaders, model architectures, and loss functions.

scripts/ – Contains the training and testing scripts, organized according to the full-data and limited-data experimental settings.

## Dataset Preparation

NN-QSM has been evaluated using QSM datasets acquired under different imaging conditions, including:

- [**SNU dataset:**](https://www.sciencedirect.com/science/article/pii/S1053811918305378?casa_token=Jr0x7xqv42cAAAAA:qkoOZpnzV1LIEriiQz0t7bnT44aAhU7vzvl0eovC13WaPi5LNqKYl34lMs2wNtC1ayvAGcDd4A) Used for training, validation, and testing.
- [**LPCNN dataset:**](https://link.springer.com/chapter/10.1007/978-3-030-59713-9_13) Used to evaluate reconstruction performance on data acquired at different field strengths and head orientations.
- [**RC-1 dataset:**](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.26830?casa_token=LjhGtm4cBSgAAAAA%3Ar_7q51Uou4qovctyNS8PQvrPEIKRNxzYO_oIu4yuArSLiSCm38mjr5g2c0woxhFsG-29y0yV-Nfgis4) Used for additional evaluation on multi-orientation data.
- [**RC-2 dataset:**](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.28716) Used for evaluation on simulated data.
- [**Hemo dataset and MS dataset:**](https://www.sciencedirect.com/science/article/pii/S1053811921006522) Clinical QSM data from patients with hemorrhage and multiple sclerosis, used to evaluate the reconstruction performance of NN-QSM in pathological conditions, including hemorrhagic abnormalities and demyelinating lesions.

Additional experiments include limited-data training and evaluation on pathological cases, including hemorrhage and multiple sclerosis.

Dataset availability, preprocessing requirements, and directory organization will be documented separately. Please refer to the original dataset publications and follow their licensing and access conditions. Raw or clinical data are not included in this repository unless redistribution is explicitly permitted.

## Training

Training instructions and the corresponding configuration files will be provided here.

The training procedure involves:

1. Preparing the local field maps and reference susceptibility maps.
2. Generating training and validation samples.
3. Configuring the network architecture, truncation depth, optimizer, and training hyperparameters.
4. Training the NN-QSM model and monitoring validation performance.
5. Saving the trained model for evaluation.

The exact training command will be added once the training script and its arguments are finalized.

## Testing and Inference

The trained model can be used to reconstruct susceptibility maps from input local field maps.

The testing instructions will cover:

1. Loading the trained model.
2. Preparing the input data and dipole kernel.
3. Performing susceptibility reconstruction.
4. Saving the reconstructed susceptibility maps.

The exact inference command and expected input/output formats will be documented alongside the testing script.

## Evaluation Metrics

Reconstruction performance can be assessed using the following metrics:

| Metric | Description |
|---|---|
| SSIM | Structural Similarity Index |
| PSNR | Peak Signal-to-Noise Ratio |
| RMSE | Root Mean Squared Error |
| HFEN | High-Frequency Error Norm |

Metric computation and comparison with reference susceptibility maps will be provided in the evaluation scripts.
