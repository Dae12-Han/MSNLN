# Multi-Scale Non-Local Attention Network
This repository is for MSNLN introduced in the following paper

[Yiqun Mei](http://yiqunm2.web.illinois.edu/), [Yuchen Fan](https://scholar.google.com/citations?user=BlfdYL0AAAAJ&hl=en), [Yuqian Zhou](https://yzhouas.github.io/), [Lichao Huang](https://scholar.google.com/citations?user=F2e_jZMAAAAJ&hl=en), [Thomas S. Huang](https://ifp-uiuc.github.io/), and [Humphrey Shi](https://www.humphreyshi.com/), "Image Super-Resolution with Cross-Scale Non-Local Attention and Exhaustive Self-Exemplars Mining", CVPR2020, [[arXiv]](https://arxiv.org/abs/2006.01424) 


The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and tested on Ubuntu 18.04 environment (Python3.6, PyTorch_1.1.0) with Titan /Xp, V100 GPUs. 
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

Our code is based on EDSR. For more information, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Run

    ```bash
    # Example X2 SR
    python3 main.py --chop --batch_size 4 --model MSNLN_631_3 --scale 2 --patch_size 64 --save MSNLN_631_3_x2 --n_feats 128 --depth 12 --data_train DIV2K --save_models

    ```

## Test
### Quick start
Pre-traind models can be downloaded from ...

    ```bash
    # No self-ensemble: CSNLN
    # Example X2 SR
    python3 main.py --model MSNLN_631_3 --data_test Set5+Set14+B100+Urban100 --data_range 801-900 --scale 2 --n_feats 128 --depth 12 --pre_train ./experiment/MSNLN_631_3_x2/model/model_best.pt --save_results --test_only --chop
    ```

## Results
### Quantitative Results
![PSNR_SSIM](/Figs/PSNR_SSIM.png)

For more results, please refer to our [paper](https://arxiv.org/abs/2006.01424) and [Supplementary Materials](http://yiqunm2.web.illinois.edu/assets/papers/04292-supp.pdf).
### Visual Results
![Visual_PSNR_SSIM](/Figs/Visual_1.png)

![Visual_PSNR_SSIM](/Figs/Visual_2.png)

![Visual_PSNR_SSIM](/Figs/Visual_3.png)


## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```

```
## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [generative-inpainting-pytorch](https://github.com/daa233/generative-inpainting-pytorch). We thank the authors for sharing their codes.

