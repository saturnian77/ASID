# ASID

Anonymous Submission

Submitted to AAAI25

## Environments
- Ubuntu 18.04
- PyTorch 2.2.2
- CUDA 9.0 & cuDNN 7.1
- Python 3.10.9

**Dependencies:**

- PyTorch>1.10
- OpenCV
- Matplotlib 3.3.4
- opencv-python
- pyyaml
- tqdm
- numpy
- torchvision

## Acknowledgement

Our codes are based on [Omni-SR](https://github.com/Francis0625/Omni-SR/)

## Guidelines for Codes

**Requisites should be installed beforehand.**

### Test

1. Make sure the location configuration is correct in ./env/env.json

2. Evaluate models with the following cmd


```
python test.py -v "Model_Name" -t tetser_Matlab -s 0 --test_dataset_name [Dataset]


[Model]: ASID_XN_DIV2K, ASIDd8_XN_DIV2K (N=2,3,4)
[Dataset]: Set5, Set14, B100, Urban100

```
Example:

```
python test.py -v "ASID_X2_DIV2K" -t tetser_Matlab -s 0 --test_dataset_name [Set5]

```

3. Execute ./PSNR_SSIM_Evaluate.m for PSNR/SSIM report. Make sure the location configuration and scale are correct in the Matlab file.
