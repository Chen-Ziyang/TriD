# :page_facing_up: Treasure in Distribution: A Domain Randomization based Multi-Source Domain Generalization for 2D Medical Image Segmentation (TriD)
This is the official pytorch implementation of our MICCAI 2023 paper "[Treasure in Distribution: A Domain Randomization based Multi-Source Domain Generalization for 2D Medical Image Segmentation](https://arxiv.org/pdf/2305.19949.pdf)". In this paper, we propose a multi-source domain generalization method called Treasure in Distribution (TriD) to construct an unprecedented search space for feature-space domain randomization.

<div align="center">
  <img width="80%" alt="TriD illustration" src="figures/Overview.png">
</div>

## Requirements
Python 3.7<br /> 
Pytorch 1.8.0

## Usage

### Installation
* Clone this repo
```
git clone https://github.com/Chen-Ziyang/TriD.git
cd TriD
```

### Data Preparation
[OD/OC Segmentation](https://zenodo.org/record/8009107)<br />
[Prostate Segmentation](https://liuquande.github.io/SAML/)

### OD/OC Segmentation
We take the scenario using BinRushed (target domain) and other four datasets (source domains) as the example.
```
cd OPTIC
# Training
CUDA_VISIBLE_DEVICES=0 python main.py --mode train_DG --mixstyle_layers layer1 layer2 --random_type TriD --Target_Dataset BinRushed --Source_Dataset Magrabia REFUGE ORIGA Drishti_GS
# Test
CUDA_VISIBLE_DEVICES=0 python main.py --mode single_test --load_time TIME_OF_MODEL --Target_Dataset BinRushed
```

### Prostate Segmentation
We take the scenario using BMC (target domain) and other five datasets (source domains) as the example.
```
cd PROSTATE
# Training
CUDA_VISIBLE_DEVICES=0 python main.py --mode train_DG --mixstyle_layers layer1 layer2 --random_type TriD --Target_Dataset BMC --Source_Dataset BIDMC HK I2CVB RUNMC UCL
# Test
CUDA_VISIBLE_DEVICES=0 python main.py --mode single_test --load_time TIME_OF_MODEL --Target_Dataset BMC
```

### Acknowledgement
Part of the code is revised from the Pytorch implementation of [DoCR](https://github.com/ShishuaiHu/DoCR).

### Citation ✏️ 📄

If you find this repo useful for your research, please consider citing the paper as follows:
```
@article{chen2023treasure,
  title={Treasure in Distribution: A Domain Randomization based Multi-Source Domain Generalization for 2D Medical Image Segmentation},
  author={Chen, Ziyang and Pan, Yongsheng and Ye, Yiwen and Cui, Hengfei and Xia, Yong},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023},
  year={2023}
}
```
