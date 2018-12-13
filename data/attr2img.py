#!/usr/bin/python
import numpy as np
from time import sleep

__ATTR__ = { 
'AwA2':	[
        'black', 'white', 'blue', 'brown', 'gray', 'orange', 'red', 'yellow', 'patches', 
        'spots', 'stripes', 'furry', 'hairless', 'toughskin', 'big', 'small', 'bulbous', 
        'lean', 'flippers', 'hands', 'hooves', 'pads', 'paws', 'longleg', 'longneck', 'tail', 
        'chewteeth', 'meatteeth', 'buckteeth', 'strainteeth', 'horns', 'claws', 'tusks', 
        'smelly', 'flys', 'hops', 'swims', 'tunnels', 'walks', 'fast', 'slow', 'strong', 
        'weak', 'muscle', 'bipedal', 'quadrapedal', 'active', 'inactive', 'nocturnal', 
        'hibernate', 'agility', 'fish', 'meat', 'plankton', 'vegetation', 'insects', 
        'forager', 'grazer', 'hunter', 'scavenger', 'skimmer', 'stalker', 'newworld', 
        'oldworld', 'arctic', 'coastal', 'desert', 'bush', 'plains', 'forest', 'fields', 
        'jungle', 'mountains', 'ocean', 'ground', 'water', 'tree', 'cave', 'fierce', 'timid', 
        'smart', 'group', 'solitary', 'nestspot', 'domestic'
        ],
'CelebA': [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs','Big_Lips', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
        'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
        'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
        'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
      	],
'RafD': [
        'pose_0', 'pose_45', 'pose_90', 'pose_135', 'pose_180',
        'neutral', 'angry', 'contemptuous', 'disgusted', 'fearful', 'happy', 'sad', 'surprised'
        ],
'painters_14': [
        'beksinski', 'boudin', 'burliuk', 'cezanne', 'chagall', 'corot', 
        'earle', 'gauguin', 'hassam', 'levitan', 'monet', 'picasso', 'ukiyoe', 'vangogh'
        ],
'WIDER': [
        'Male', 'longHair', 'sunglass', 'Hat', 'Tshiirt', 'longSleeve', 'formal', 'shorts',
        'jeans', 'longPants', 'skirt', 'faceMask', 'logo', 'stripe'
        ],

'Animals': [
        'antelope', 'grizzly+bear', 'killer+whale', 'beaver', 'dalmatian', 'persian+cat', 'horse', 
        'german+shepherd', 'blue+whale', 'siamese+cat', 'skunk', 'mole', 'tiger', 'hippopotamus', 
        'leopard', 'moose', 'spider+monkey', 'humpback+whale', 'elephant', 'gorilla', 'ox', 'fox', 
        'sheep', 'seal', 'chimpanzee', 'hamster', 'squirrel', 'rhinoceros', 'rabbit', 'bat', 'giraffe', 
        'wolf', 'chihuahua', 'rat', 'weasel', 'otter', 'buffalo', 'zebra', 'giant+panda', 'deer', 
        'bobcat', 'pig', 'lion', 'mouse', 'polar+bear', 'collie', 'walrus', 'raccoon', 'cow', 'dolphin'
        ],    
'Image2Weather': [
        'cloudy', 'foggy', 'rain', 'snow', 'sunny'
        ],
'BAM': [
        'content_bicycle', 'content_bird', 'content_building', 'content_cars', 
        'content_cat', 'content_dog', 'content_flower', 'content_people', 'content_tree', 
        'emotion_gloomy', 'emotion_happy', 'emotion_peaceful', 'emotion_scary', 
        'media_3d_graphics', 'media_comic', 'media_graphite', 'media_oilpaint', 
        'media_pen_ink', 'media_vectorart', 'media_watercolor'      
        ],
'Image2Season': [
        'autumn', 'spring', 'summer', 'winter'
        ]
}

# Birds = [line.strip().split(' ')[1].replace('_','@').replace('::','_') for line in open('data/Birds/CUB_200_2011/attributes.txt').readlines()]
# birds = []
# for line in Birds:
#   _str = ''.join([l.capitalize() for l in line.split('@')])
#   birds.append(_str)
# __ATTR__['Birds']= birds

def replace_break_line(text):
  if '_o_' in text: text = text.replace('_o_', '_o ')
  if '_' in text: text = text.replace('_', '\n')
  if '+' in text: text = text.replace('+', '\n')
  text = text.split('\n')
  return text  

def get_max_size(FONT, text, base_size):
  max_size = 70
  font = FONT(max_size)
  while (font.getsize(text)[0]>=base_size):
    max_size-=1
    font = FONT(max_size) 
    textsize = font.getsize(text)
  return max_size

def get_font():
  from PIL import ImageFont
  return lambda size: ImageFont.truetype("data/Times-Roman.otf", size)

def get_img(text, background='white', size=None):
  import numpy as np, ipdb
  from PIL import ImageFont, ImageDraw, Image
  base_size = (256,256)
  foreground = (0,0,0) if background=='white' else (255,255,255)
  background = (255,255,255) if background=='white' else (0,0,0)
  img = Image.new('RGB', base_size, background)
  
  text = replace_break_line(text)
  text = [line.capitalize() for line in text]

  draw = ImageDraw.Draw(img)
  FONT = get_font()
  if size is None: 
    size = []
    for idx, _text in enumerate(text):
      size.append(get_max_size(FONT, _text, base_size[0]))
    size = min(size)    
  
  font = FONT(size)
  previous_y = 0
  for _text in text[::-1]:
    _text = _text
    textsize = font.getsize(_text)
    textX = (img.size[0] - textsize[0]) / 2
    textY = img.size[1] - textsize[1] - previous_y - 5
    draw.text((textX, textY ), _text, font=font, fill=foreground)
    previous_y += textsize[1]
  return img

def external2img(attributes, img_size):
  FONT = get_font()
  size = []
  for idx, attr in enumerate(attributes):
    text = replace_break_line(attr)
    text = [line.capitalize() for line in text]   
    for _text in text: 
      size.append(get_max_size(FONT, _text, img_size))
  size = min(size)     
  return text2img(attributes, size=size)

def text2img(attributes, save=None, size=None):
  import ipdb
  # ipdb.set_trace()
  assert type(attributes)==list
  imgs=[]
  for idx, attr in enumerate(attributes):
    color='white'#'black' if attr in ['Source', 'Off'] else 'white'
    text = attr.capitalize()# if dataset!= 'Birds' else attr
    img = get_img(text, color, size=size)
    imgs.append(img)
    if save is not None: 
      path = os.path.join(save, '{}.jpeg'.format(attr))
      img.save(save)  
  if save is None: return imgs

if __name__ == '__main__':
  import argparse, os
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='all')
  opt = parser.parse_args()
  if opt.dataset!='all':
    assert os.path.isdir(opt.dataset)
    opt.dataset = [opt.dataset]
  else:
    opt.dataset = __ATTR__.keys()

  for dataset in opt.dataset:
    folder = os.path.join(dataset, 'aus_flat')
    if not os.path.isdir(folder): os.makedirs(folder)
    __ATTR__[dataset] = ['Source', 'Off'] + __ATTR__[dataset]
    text2img(__ATTR__[dataset], save=folder)

