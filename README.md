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
NN-QSM/
├── README.md
├── requirements.txt
├── models/          # Network architectures
├── utils/           # Dipole operators and utility functions
├── scripts/         # Training, testing, and evaluation
├── configs/         # Experimental configurations
├── data/            # Dataset instructions (not raw data)
├── checkpoints/     # Pretrained model instructions
└── figures/         # Method overview and results
```


## Dataset Preparation

NN-QSM has been evaluated using QSM datasets acquired under different imaging conditions, including:

- **SNU dataset:** Used for training, validation, and testing.
- **LPCNN dataset:** Used to evaluate reconstruction performance on data acquired at different field strengths and head orientations.
- **RC-1 dataset:** Used for additional evaluation on multi-orientation data.
- **RC-2 dataset:** Used for evaluation on simulated data.
- **Hemo dataset:**
- **MS dataset:**

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
