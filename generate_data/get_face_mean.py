import numpy as np
import imageio
import ipdb
import glob
import matplotlib.pyplot as plt
import os
import tqdm
import pickle

mode_data = 'normal'
stats_folder = '../data/stats'
txt_files = glob.glob('../data/MultiLabelAU/{}/*/test.txt'.format(mode_data))
lines = []

for txt in txt_files:
	# ipdb.set_trace()
	lines.extend([i.split(' ')[0] for i in open(txt).readlines() if os.path.isfile(i.split(' ')[0])])
lines = list(set(sorted(lines)))
print("Calculating Statistics from {} images...".format(len(lines)))

name = 'Faces_aligned' if 'aligned' in txt_files[0] else 'Faces'

mean=0.0
std_face=0.0
std_channels=0.0
for line in tqdm.tqdm(lines, desc='Calculating MEAN'):
	line = line.replace(name, name+'_'+'256')
	img = imageio.imread(line).astype(np.float64)
	mean += ( img / float(len(lines)) )

for line in tqdm.tqdm(lines, desc='Calculating STD'):
	line = line.replace(name, name+'_'+'256')
	img = imageio.imread(line).astype(np.float64)
	std_face += ( ((img - mean)**2) / float(len(lines)) )	
	std_channels += ( ((img - mean)**2).sum(axis=(0,1)) / float(len(lines)*img.shape[0]*img.shape[1]) )	

std_face = (np.sqrt(std_face )).astype(np.float64)
std_channels = (np.sqrt(std_channels )).astype(np.float64)
# ipdb.set_trace()

##MEAN
mean_img = stats_folder+'/face_{}_mean.jpg'.format(mode_data)
np.save(mean_img.replace('jpg','npy'), mean_img)
img = mean.astype(np.uint8)
imageio.imwrite(mean_img, img)

mean_channels = stats_folder+'/channels_{}_mean.npy'.format(mode_data)
np.save(mean_channels, mean.mean(axis=(0,1)))

##STD
std_img = stats_folder+'/face_{}_std.jpg'.format(mode_data)
np.save(std_img.replace('jpg','npy'), std_face)
img = std_face.astype(np.uint8)
imageio.imwrite(std_img, img)

std_channels_file = stats_folder+'/channels_{}_std.npy'.format(mode_data)
np.save(std_channels_file, std_channels)

# ipdb.set_trace()