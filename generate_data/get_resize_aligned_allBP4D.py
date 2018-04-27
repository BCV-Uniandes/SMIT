import sys
import pickle
import os
import cv2 as cv
import numpy as np
import random
import ipdb
import matlab.engine
import matlab
import time
import datetime
import tqdm
import glob
from shutil import copyfile

class Get_Faces():
    def __init__(self):
        pwd = os.getcwd()
        os.chdir('tools')
        from get_faces import __init__, face_from_file, imshow
        net_face = __init__()
        os.chdir(pwd)
        self.net = net_face
        self.face_from_file = face_from_file
        self.imshow = imshow

    def from_file(self, img_file):
        return self.face_from_file(self.net, img_file)


def display_time(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    string_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)
    print("Time elapsed: "+string_time)   
    print(datetime.datetime.now())  
    print(" ")

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
    parser.add_argument('--img_size', type=int, default=256, help='size of the image to resize')
    parser.add_argument('--gpu', type=str, default='2', help='GPU device')
    parser.add_argument('--aligned', action='store_true', default=False)
    parser.add_argument('--OF', action='store_true', default=False)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

    folder_root = '/home/afromero/datos/Databases/BP4D'
    org_root = os.path.join(folder_root,'Sequences')
    # face_root = '/home/afromero/Codes/ActionUnits/data/Faces/BP4D/Sequences'
    face_root = os.path.join(folder_root, 'Faces')
    algined_root = os.path.join(folder_root, 'Faces_aligned')
    algined_size_root = os.path.join(folder_root,'Faces_aligned_{}'.format(args.img_size))
    faces_size_root = os.path.join(folder_root,'Faces_{}'.format(args.img_size))

    if not os.path.isdir(face_root): os.makedirs(face_root)  
    if not os.path.isdir(algined_root): os.makedirs(algined_root)  
    if not os.path.isdir(algined_size_root): os.makedirs(algined_size_root)  

    pwd = os.getcwd()
    os.chdir('/home/afromero/Codes/Face_Alignment/MTCNN_face_detection_alignment/code/codes/MTCNNv2')
    future = matlab.engine.connect_matlab(async=True)
    eng = future.result()
    eng = matlab.engine.start_matlab()
    os.chdir(pwd)

    # ipdb.set_trace()
    img_files = sorted(glob.glob(org_root+'/*/*/*.jpg'))
    txt_file = os.path.join(folder_root, 'data.txt')
    f = open(txt_file, 'w')
    for im in img_files:
        f.writelines(os.path.abspath(im)+'\n')
    f.close()
    # ipdb.set_trace()

    org_files = [line.strip() for line in open(txt_file).readlines()]
    # ipdb.set_trace()
    Faces = Get_Faces()
    face_files = []
    count = 0
    # ipdb.set_trace()        
    for org_file in tqdm.tqdm(org_files, total=len(org_files), \
                desc='Extracting Faces', ncols=80, leave=True):
        file_name = '/'.join(org_file.split('/')[-3:])
        # folder_name = os.path.dirname(org_file)
        # face_name = file_name.split('.')[0]+'_Faces.'+file_name.split('.')[1]
        face_file = os.path.join(face_root, file_name)
        face_dir = os.path.dirname(face_file)
        if not os.path.isdir(face_dir): os.makedirs(face_dir)
        if not os.path.isdir(face_dir.replace(face_root, algined_root)): os.makedirs(face_dir.replace(face_root, algined_root))
        if args.OF:
            img_face, bbox = Faces.from_file(org_file, BBOX=True)
            of_file = org_file.replace('Sequences', 'Sequences_Flow')
            of_face_file = face_file.replace('Sequences', 'Sequences_Flow')
            if not os.path.isfile(of_face_file):
                try: 
                    of_face = imageio.imread(of_file)[bbox]
                    imageio.imwrite(face_file, of_face)
                except: 
                    continue#ipdb.set_trace()                

        else:
            if os.path.isfile(face_file): 
                face_files.append(face_file)
                continue            
            img_face = Faces.from_file(org_file)
        # Faces.imshow(img_face)
        try: imageio.imwrite(face_file, img_face[:,:,::-1])
        except: continue#ipdb.set_trace()
        face_files.append(face_file)

    if args.aligned: 
        print(' [*] Performing alignment...')
        _f = os.path.abspath(txt_file.replace('.txt', '_faces.txt'))
        f = open(_f, 'w')
        for rs_file in face_files: 
            # if os.path.isfile(rs_file.replace('Faces', 'Faces_aligned')):
            #     os.remove(rs_file.replace('Faces', 'Faces_aligned'))
            f.writelines(rs_file+'\n')
        f.close()
        # ipdb.set_trace()
        #_ = eng.face_alignment(_f)
        # os.remove(_f)
        print(' [°] Alignment done')  

        _f = os.path.abspath(txt_file.replace('.txt', '_faces_aligned_{}.txt'.format(args.img_size)))
        f = open(_f, 'w')           
        for rs_file in tqdm.tqdm(face_files, total=len(face_files), \
                    desc='Resizing Facial Alignment to %d'%(args.img_size), ncols=80, leave=True):             
            file_aligned = rs_file.replace('Faces', 'Faces_aligned')
            if not os.path.isfile(file_aligned): continue
            file_name = '/'.join(rs_file.split('/')[-3:])
            resized_file = os.path.join(algined_size_root, file_name)
            get_resize(file_aligned, resized_file, args.img_size)
            f.writelines(resized_file+'\n')
        f.close()  

    else:
        _f = os.path.abspath(txt_file.replace('.txt', '_faces_{}.txt'.format(args.img_size)))
        f = open(_f, 'w')           
        for rs_file in tqdm.tqdm(face_files, total=len(face_files), \
                    desc='Resizing Faces to %d'%(args.img_size), ncols=80, leave=True):             
            if not os.path.isfile(rs_file): continue
            file_name = '/'.join(rs_file.split('/')[-3:])            
            resized_file = os.path.join(faces_size_root, file_name)
            if args.OF:
                resize_file_of = resized_file.replace('Sequences', 'Sequences_Flow')
                rs_file_of = rs_file.replace('Sequences', 'Sequences_Flow')
                get_resize(rs_file_of, resize_file_of, args.img_size)
            get_resize(rs_file, resized_file, args.img_size)
            f.writelines(resized_file+'\n')
        f.close()  


