# Multi-Scale Non-Local Attention Network
This repository is for MSNLN introduced in the following paper ...  
Sowon Kim, Hanhoon Park*, "Super-Resolution Using Multi-Scale Non-Local Attention", 2022  
The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [CSNLN](https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention), tested on window 10 environment (Python3.6.5, PyTorch_1.7.1) with GeForce RTX 2080 Ti GPU. 

## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction

....

![CS-NL Attention](/Figs/Attention.png)

Cross-Scale Non-Local Attention.

![CSNLN](/Figs/CSNLN.png)

The recurrent architecture with Self-Exemplars Mining (SEM) Cell.

## Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Specify '--dir_data' based on the HR and LR images path. 

Code is based on EDSR and CSNLN. For more information, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) or [CSNLN](https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention).

### Run

    ```
    # Example X2 SR
    python3 main.py --chop --batch_size 4 --model MSNLN_631_3 --scale 2 --patch_size 64 --save MSNLN_631_3_x2 --n_feats 128 --depth 12 --data_train DIV2K --save_models
    ```

## Test
### Quick start
Pre-traind models can be downloaded from ...

    ```
    # No self-ensemble: MSNLN
    # Example X2 SR
    python3 main.py --model MSNLN_631_3 --data_test Set5+Set14+B100+Urban100 --data_range 801-900 --scale 2 --n_feats 128 --depth 12 --pre_train ./experiment/MSNLN_631_3_x2/model/model_best.pt --save_results --test_only --chop
    ```

## Results
### Quantitative Results
![PSNR_SSIM](/Figs/Table1.png)

![PSNR_SSIM](/Figs/Table2.png)

### Visual Results
![Visual_PSNR_SSIM](/Figs/Fig1.png)

![Visual_PSNR_SSIM](/Figs/Visual_2.png)

![Visual_PSNR_SSIM](/Figs/Visual_3.png)


## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```

```
## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [CSNLN](https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention). We thank the authors for sharing their codes.

