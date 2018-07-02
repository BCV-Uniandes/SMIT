#!/usr/bin/ipython
import os
import argparse
from data_loader import get_loader
import math
import ipdb
from torchvision.utils import save_image
import warnings
import ipdb
warnings.filterwarnings('ignore')

def denorm(x):
  out = (x + 1) / 2
  return out.clamp_(0, 1)

def main(config):
  img_size = config.image_size

  data_loader = get_loader(config.metadata_path, img_size,
                   img_size, config.batch_size, 'MultiLabelAU', config.mode)  

  data = next(iter(data_loader))[0]
  data = denorm(data.cpu())
  print("Saved at: {}/sample.jpg".format(config.metadata_path))
  save_image(data.cpu(), '{}/sample.jpg'.format(config.metadata_path),nrow=int(math.sqrt(data.size(0))), padding=0)  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # Model hyper-parameters
  parser.add_argument('--image_size', type=int, default=128)

  # Training settings
  parser.add_argument('--dataset', type=str, default='MultiLabelAU', choices=['MultiLabelAU', 'EmotionNet'])
  parser.add_argument('--batch_size', type=int, default=49)

  # Misc
  parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'])

  # Path
  parser.add_argument('--fold', type=str, default='0')
  parser.add_argument('--mode_data', type=str, default='normal', choices=['normal', 'aligned'])  
  parser.add_argument('--metadata_path', type=str, default='./data/MultiLabelAU')


  config = parser.parse_args()
  config.metadata_path = os.path.join(config.metadata_path, config.mode_data, 'fold_'+str(config.fold))
  config.metadata_path = config.metadata_path.replace('MultiLabelAU', config.dataset)

  main(config)