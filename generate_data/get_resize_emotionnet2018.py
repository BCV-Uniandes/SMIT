import sys
import pickle
import os
import numpy as np
import random
import ipdb
import tqdm
import glob


def get_resize(org_file, resized_file, img_size, OF=False):
    import skimage.transform
    import imageio
    if type(img_size)==int: img_size = [img_size, img_size]
    elif type(img_size)==list and len(img_size)==1: img_size = [img_file[0], img_size[0]]
    folder = os.path.dirname(resized_file)
    if not os.path.isdir(folder): os.makedirs(folder) 
    # ipdb.set_trace()
    imageio.imwrite(resized_file, \
        (skimage.transform.resize(imageio.imread(org_file), (img_size[0], img_size[1]))*255).astype(np.uint8))
  
if __name__ == '__main__':    
    import argparse
    import imageio

    parser = argparse.ArgumentParser(description='Txt file with path_to_image and 12 different AUs to LMDB')
    parser.add_argument('--img_size', type=int, default=128, help='size of the image to resize')
    args = parser.parse_args()
    modes = ['train', 'val']
    folder_root = '/home/afromero/ssd2/EmotionNet2018/data'
    # face_root = '/home/afromero/Codes/ActionUnits/data/Faces/BP4D/Sequences'
    size_root = folder_root.replace(os.path.basename(folder_root), os.path.basename(folder_root)+'_'+str(args.img_size))

    if not os.path.isdir(size_root): os.makedirs(size_root)  

    # ipdb.set_trace()
    img_files = sorted([glob.glob(folder_root+'/'+mode+'/*.jpg') for mode in modes])

    # ipdb.set_trace()  
    for idx, mode in enumerate(modes):   
        for org_file in tqdm.tqdm(img_files[idx], total=len(img_files[idx]), \
                desc='Resizing Files [%s]'%(mode), ncols=80, leave=True):
            resize_file = org_file.replace(folder_root, size_root)
            # ipdb.set_trace() 
            get_resize(org_file, resize_file, args.img_size)