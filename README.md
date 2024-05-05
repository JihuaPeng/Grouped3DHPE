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
python run_onestage.py -k cpn_ft_h36m_dbb --stage 1 -lfd 512 -e 80
```
### Three-stage strategy

For the first stage, run:

```bash
python run_threestage.py -k cpn_ft_h36m_dbb --stage 1 -lfd 512 -e 80
```

For the second stage, run:
```bash
python run_threestage.py -k cpn_ft_h36m_dbb --stage 2 -lfd 512 -p stage_1_best_model.bin -e 80
```

For the third stage, run:
```bash
python run_threestage.py -k cpn_ft_h36m_dbb --stage 3 -lfd 512 -ft stage_2_best_model.bin -lr 0.0005 -e 80
```
## Evaluating our models

You can download our pre-trained models from [Google Drive](https://drive.google.com/drive/u/0/my-drive). Put `CPN/cpn_one-stage_best_epoch.bin`, `CPN/cpn_three-stage_3_best_epoch.bin`, `GT/gt_one-stage_best_epoch.bin` and `GT/gt_three-stage_3_best_epoch.bin` in the `./checkpoint` directory. Both of the models are trained on Human3.6M dataset.

