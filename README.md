
# Stochastic Multi-Label Image-to-image Translation 

## Train command:
```bash
./main.py -- --GPU=0 --dataset_fake=CelebA --GAN_options=RaGAN,Stochastic,AdaIn2,Split_Optim,InterStyleConcatLabels,HINGE,SpectralNorm,Attention,LayerNorm \
		--mode_data=normal --image_size=256 --batch_size=16 --style_dim=20 --d_train_repeat=1 --ALL_ATTR=4 --lambda_style=0 --lambda_mask=0.1 --MultiDis=3
# ./main.py -- --GPU... the double line in between is because IPython. Remove for python execution. 
```

#### GAN_options
SMIT default options are: 
- Stochastic
- AdaIn2 (uses all residuals layers as AdaIN)
- Split_Optim (fixed weights)
- InterStyleConcatLabels (introduces both label and style through AdaIN)
- LayerNorm (normalization layer for upsampling. 

Other options improves SMIT: 
- RaGAN (stabilization technique)
- HINGE (loss) 
- SpectralNorm (normalization layer for the discriminator)
- Attention (attention loss)

#### mode_data
It uses both, *faces* and *normal*. *faces* does not crop and just resize, *normal* randomly crop and resizes instead.

#### style_dim
All experiments were with stye_dim=20

#### lambda_style
This is probably one of the most important. It **must** be zero (0). Otherwise, SMIT will deploy a style encoder. 

#### MultiDis
Uses MultiDiscriminator by means of downsampling the input N times. It is the same whether this set to zero or one. 

#### lambda_mask
Experiments that involves small changes (facial expressions and facial analsis) it is set to 0.1. Otherwise, 1.0.

#### ALL_ATTR
This flag is optional. It only modifies the data loader. 
