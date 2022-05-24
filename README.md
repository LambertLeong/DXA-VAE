# DXA-VAE

This repo houses code used for NeurIPS 2022 submission entitled "Quantitative Imaging Principles Improves Medical Image Learning"

- DXA = Dual Energy X-ray Absorptiometry
- VAE = Variational AutoEncoder

---

## Prerequisites

- Linux Kernel: 4.4+
- CUDA Version: 10.1
- Python 3.6+ and the following modules:
  - Keras_Preprocessing==1.1.2
  -  matplotlib==3.3.4
  -  numpy==1.19.2
  -  pandas==1.2.3
  -  tensorflow==1.13.1

## Usage

- In \*.config file:
  - define path to directory containing data split csv files  
    - directory should contain a files labeled "train.csv", "val.csv", and "test.csv"
      - csv files should contain on column and each row contains a path to DXA npy files
  - define model parameters
  - specify path for saved model
- run train.py

### Example
```sh
python train.py -c train.config
```
---
## Datasets

Data used for training and experiments are available upon request. Inquire at
[https://shepherdresearchlab.org/services/](https://shepherdresearchlab.org/services/)

---
## Models and Weights

Trained VAE, encoder, and Pseudo-DXA models are available upon request. Inquire at
[https://shepherdresearchlab.org/services/](https://shepherdresearchlab.org/services/)
