[![Build Status](https://dev.azure.com/bcv-uniandes/BCV%20Public%20Repos/_apis/build/status/BCV-Uniandes.SMIT?branchName=master)](https://dev.azure.com/bcv-uniandes/BCV%20Public%20Repos/_build/latest?definitionId=19&branchName=master)

# **SMIT**: Stochastic Multi-Label Image-to-image Translation 

This repository provides a PyTorch implementation of [SMIT](https://arxiv.org/abs/1812.03704). SMIT can stochastically translate an input image to multiple domains using only a single generator and a discriminator. It only needs a target domain (binary vector e.g., [0,1,0,1,1] for 5 different domains) and a random gaussian noise. 
<br/>

## Paper
[SMIT: Stochastic Multi-Label Image-to-image Translation ](https://arxiv.org/abs/1812.03704) <br/>
[Andrés Romero](https://afromero.co/en)<sup> 1</sup>, [Pablo Arbelaez](https://scholar.google.com.co/citations?user=k0nZO90AAAAJ&hl=en)<sup>1</sup>, [Luc Van Gool](https://scholar.google.com/citations?user=TwMib_QAAAAJ&hl=en)<sup> 2</sup>, [Radu Timofte](https://scholar.google.com/citations?user=u3MwH5kAAAAJ&hl=en)<sup> 2</sup> <br/>
<sup>1 </sup>Biomedical Computer Vision ([BCV](https://biomedicalcomputervision.uniandes.edu.co/)) Lab, Universidad de Los Andes. <br/>
<sup>2 </sup>Computer Vision Lab ([CVL](https://www.vision.ee.ethz.ch/en/)), ETH Zürich. <br/>
<br/>

<p align="center"><img width="25%" src="Figures/andres.jpg" /><img width="25%" src="Figures/pablo.jpg" /><img width="25%" src="Figures/luc.jpg" /><img width="25%" src="Figures/radu.jpg" /></p>

## Citation
```
@article{romero2019smit,
  title={SMIT: Stochastic Multi-Label Image-to-Image Translation},
  author={Romero, Andr{\'e}s and Arbel{\'a}ez, Pablo and Van Gool, Luc and Timofte, Radu},
  journal={ICCV Workshops},
  year={2019}
}
```
<br/>

## Dependencies
* [Python](https://www.continuum.io/downloads) (2.7, 3.5+)
* [PyTorch](http://pytorch.org/) (0.3, 0.4, 1.0)
<br/>

## Usage

### Cloning the repository
```bash
$ git clone https://github.com/BCV-Uniandes/SMIT.git
$ cd SMIT
```

### Downloading the dataset
To download the CelebA dataset:
```bash
$ bash generate_data/download.sh
```

### Train command:
```bash
./main.py --GPU=$gpu_id --dataset_fake=CelebA
```
Each dataset must has `datasets/<dataset>.py` and `datasets/<dataset>.yaml` files. All models and figures will be stored at `snapshot/models/$dataset_fake/<epoch>_<iter>.pth` and `snapshot/samples/$dataset_fake/<epoch>_<iter>.jpg`, respectivelly.

### Test command:
```bash
./main.py --GPU=$gpu_id --dataset_fake=CelebA --mode=test
```
SMIT will expect the `.pth` weights are stored at `snapshot/models/$dataset_fake/` (or --pretrained_model=location/model.pth should be provided). If there are several models, it will take the last alphabetical one. 

### Demo:
```bash
./main.py --GPU=$gpu_id --dataset_fake=CelebA --mode=test --DEMO_PATH=location/image_jpg/or/location/dir
```
DEMO performs transformation per attribute, that is swapping attributes with respect to the original input as in the images below. Therefore, *--DEMO_LABEL* is provided for the real attribute if *DEMO_PATH* is an image (If it is not provided, the discriminator acts as classifier for the real attributes).

### [Pretrained models](http://marr.uniandes.edu.co/weights/SMIT)
Models trained using Pytorch 1.0.

### Multi-GPU
For multiple GPUs we use [Horovod](https://github.com/horovod/horovod). Example for training with 4 GPUs:
```bash
mpirun -n 4 ./main.py --dataset_fake=CelebA
```
<br/>

## Qualitative Results. Multi-Domain Continuous Interpolation.
First column (original input) -> Last column (Opposite attributes: smile, age, genre, sunglasses, bangs, color hair). Up: Continuous interpolation for the fake image. Down: Continuous interpolation for the attention mechanism.

<p align="center"><img width="100%" src="Figures/interpolation.jpg"/></p>
<p align="center"><img width="100%" src="Figures/interpolation_attn.jpg"/></p>

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
