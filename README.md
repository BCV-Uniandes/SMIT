[![Build Status](http://marr.uniandes.edu.co/api/badges/BCV-Uniandes/SMIT/status.svg)](http://marr.uniandes.edu.co/BCV-Uniandes/SMIT)

# **SMIT**: Stochastic Multi-Label Image-to-image Translation 

This repository provides a PyTorch implementation of [SMIT](https://arxiv.org/abs/1812.03704). SMIT can stochastically translate an input image to multiple domains using only a single generator and a discriminator. It only needs a target domain (binary vector e.g., [0,1,0,1,1] for 5 different domains) and a random gaussian noise. 

<br/>

## Paper
[SMIT: Stochastic Multi-Label Image-to-image Translation ](https://arxiv.org/abs/1711.09020) <br/>
[Andrés Romero](https://afromero.co/en)<sup> 1,2</sup>, [Pablo Arbelaez](https://scholar.google.com.co/citations?user=k0nZO90AAAAJ&hl=en)<sup>1</sup>, [Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en)<sup> 2</sup>, [Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en)<sup> 2</sup> <br/>
<sup>1 </sup>Biomedical Computer Vision ([BCV](https://biomedicalcomputervision.uniandes.edu.co/)) Lab, Universidad de Los Andes. <br/>
<sup>2 </sup>Computer Vision Lab ([CVL](https://www.vision.ee.ethz.ch/en/)), ETH Zürich. <br/>

<br/>

## Dependencies
* [Python](https://www.continuum.io/downloads) (2.7, 3.5+)
* [PyTorch](http://pytorch.org/) (0.3, 0.4, 1.0)

<br/>

## Usage

### 1. Cloning the repository
```bash
$ git clone https://github.com/BCV-Uniandes/SMIT.git
$ cd SMIT
```

### 2. Downloading the dataset
To download the CelebA dataset:
```bash
$ bash generate_data/download.sh
```

### Train command:
```bash
./main.py --GPU=$gpu_id --dataset_fake=CelebA
```
Each dataset has a `datasets/<dataset>.py` and a `datasets/<dataset>.yaml` files. All models and figures will be stored at `snapshot/models/$dataset_fake/<epoch>_<iter>.pth` and `snapshot/samples/$dataset_fake/<epoch>_<iter>.jpg`, respectivelly.

### Test command:
```bash
./main.py --GPU=$gpu_id --dataset_fake=CelebA --mode=test
```
SMIT will expect the `.pth` weights are stored at `snapshot/models/$dataset_fake/`. If there are several models, it will take the last alphabetical one. 

<br/>

## Qualitative Results. Random sampling. 

### CelebA
![](Figures/CelebA_Multimodal_Random.jpg)

![](Figures/CelebA_Attention_Multimodal_Random.jpg)

### EmotionNet
![](Figures/EmotionNet_Multimodal_Random.jpg)

![](Figures/EmotionNet_Attention_Multimodal_Random.jpg)

### RafD
![](Figures/RafD_Multimodal_Random.jpg)

![](Figures/RafD_Attention_Multimodal_Random.jpg)

### Edges2Shoes
![](Figures/Shoes_Multimodal_Random.jpg)

### Edges2Handbags
![](Figures/Handbags_Multimodal_Random.jpg)

### Yosemite
![](Figures/Yosemite_Multimodal_Random.jpg)

### Painters
![](Figures/Painters_Multimodal_Random.jpg)

<br/>

## Qualitative Results. Style Interpolation between first and last row.

### CelebA
![](Figures/CelebA_Multimodal_Interpolation.jpg)

### EmotionNet
![](Figures/EmotionNet_Multimodal_Interpolation.jpg)

### RafD
![](Figures/RafD_Multimodal_Interpolation.jpg)

### Edges2Shoes
![](Figures/Shoes_Multimodal_Interpolation.jpg)

### Edges2Handbags
![](Figures/Handbags_Multimodal_Interpolation.jpg)

### Yosemite
![](Figures/Yosemite_Multimodal_Interpolation.jpg)

### Painters
![](Figures/Painters_Multimodal_Interpolation.jpg)

<br/>

## Qualitative Results. Label continuous inference between first and last row.

### CelebA
![](Figures/CelebA_Label_Interpolation.jpg)

### EmotionNet
![](Figures/EmotionNet_Label_Interpolation.jpg)