# EHFusion: An efficient heterogeneous fusion model for group-based 3D human pose estimation

This is the readme file for the code release of "EHFusion: An efficient heterogeneous fusion model for group-based 3D human pose estimation" on PyTorch platform.

## Dependencies
Make sure you have the following dependencies installed:
* PyTorch >= 0.4.0
* NumPy
* Matplotlib=3.1.0

## Dataset

Our model is evaluated on [Human3.6M](http://vision.imar.ro/human3.6m) and [HumanEva-I](http://humaneva.is.tue.mpg.de/datasets_human_1) datasets.

### Human3.6M
We set up the Human3.6M dataset in the same way as [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md). 

### HumanEva-I
We set up the HumanEva-I dataset in the same way as [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md).

## Training from scratch
### One-stage strategy

```bash
python run_onestage.py -k gt --stage 1 -lfd 256
