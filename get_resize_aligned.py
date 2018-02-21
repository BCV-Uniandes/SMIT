import sys
import pickle
import os
import cv2 as cv
import numpy as np
import random
import ipdb
import time
import os
import matlab.engine
import matlab
import time
import datetime
import tqdm
import ipdb

def display_time(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    string_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    print("Time elapsed: "+string_time)   
    print(datetime.datetime.now())  
    print(" ")

def get_resize(org_file, resized_file, img_size):
    import skimage.transform
    import imageio
    folder = os.path.dirname(resized_file)
    if not os.path.isdir(folder): os.makedirs(folder) 
    imageio.imwrite(resized_file, skimage.transform.resize(imageio.imread(org_file), (img_size, img_size)).astype(np.uint8))

pwd = os.getcwd()
os.chdir('/home/afromero/Codes/Face_Alignment/MTCNN_face_detection_alignment/code/codes/MTCNNv2')
future = matlab.engine.connect_matlab(async=True)
eng = future.result()
eng = matlab.engine.start_matlab()
os.chdir(pwd)
    
if __name__ == '__main__':    
    import argparse

    parser = argparse.ArgumentParser(description='Txt file with path_to_image and 12 different AUs to LMDB')
    parser.add_argument('--mode', type=str, default='Training', help='Mode: Training/Test (default: Training)')
    parser.add_argument('--img_size', type=int, default=256, help='size of the image to resize')
    parser.add_argument('--fold', type=str, default='all', help='fold crossvalidation')
    parser.add_argument('--aligned', action='store_true', default=False)


    args = parser.parse_args()

    if args.fold == 'all':
        folds = [0,1,2]
    else:
        folds = [args.fold]

    for fold in folds:
        fold = int(fold)

        txt_file  = 'data/MultiLabelAU/aligned/fold_{}/{}.txt'.format(fold, args.mode)

        org_files = [line.split()[0] for line in open(txt_file).readlines()]

        resized_files = []
        count = 0
        # ipdb.set_trace()        
        for file_ in tqdm.tqdm(org_files, total=len(org_files), \
                    desc='Resizing - fold %d'%(fold), ncols=80, leave=True):
            org_file = file_.replace('BP4D_256', 'BP4D')
            org_file = org_file.replace('Faces_aligned', 'Faces')
            resized_file = org_file.replace('BP4D', 'BP4D_'+str(args.img_size))

            if not os.path.isfile(resized_file): get_resize(org_file, resized_file, args.img_size)
            resized_files.append(resized_file)


        if args.aligned: 
            print(' [*] Performing alignment...')
            _f = 'temp_txt'
            f = open(_f, 'w')
            for rs_file in resized_files: f.writelines(rs_file)
            f.close()
            _ = eng.face_alignment(aligned_file)
            os.remove(_f)
            print(' [°] Alignment done')