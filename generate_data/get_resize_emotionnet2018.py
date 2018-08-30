import sys
import pickle
import os
import numpy as np
import random
import ipdb
import tqdm
import glob
import imageio
import skimage.transform

def imshow(im_file, resize=0):
    import matplotlib.pyplot as plt
    img = imageio.imread(im_file)
    if resize: img = (skimage.transform.resize(img, (resize, resize))*255).astype(np.uint8)
    plt.imshow(img)
    plt.show()


def get_resize(org_file, resized_file, img_size, OF=False):
    if type(img_size)==int: img_size = [img_size, img_size]
    elif type(img_size)==list and len(img_size)==1: img_size = [img_file[0], img_size[0]]
    folder = os.path.dirname(resized_file)
    if not os.path.isdir(folder): os.makedirs(folder) 
    # ipdb.set_trace()
    imageio.imwrite(resized_file, \
        (skimage.transform.resize(imageio.imread(org_file), (img_size[0], img_size[1]))*255).astype(np.uint8))
  
class Face():
    def __init__(self):
        os.chdir('DFace')
        from dface.core.detect import create_mtcnn_net, MtcnnDetector
        import numpy as np
        pnet, rnet, onet = create_mtcnn_net(p_model_path="./model_store/pnet_epoch.pt", r_model_path="./model_store/rnet_epoch.pt", o_model_path="./model_store/onet_epoch.pt", use_cuda=False)
        self.detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=64)
        os.chdir('..')

    def get_face_from_file(self, org_file):
        img = imageio.imread(org_file)       
        try: 
            bbox = map(int, self.detector.detect_face(img[:,:,::-1])[0][0][:-1])
            bbox = [max(0,i) for i in bbox]
            img_face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if 0 in img_face.shape:
                ipdb.set_trace()
        except:
            img_face = img
        return img_face

    def get_face_and_save(self, org_file, face_file):
        face = self.get_face_from_file(org_file)
        imageio.imwrite(face_file, face)    

if __name__ == '__main__':    
    import argparse
    import imageio

    parser = argparse.ArgumentParser(description='Txt file with path_to_image and 12 different AUs to LMDB')
    parser.add_argument('--img_size', type=int, default=128, help='size of the image to resize')
    parser.add_argument('--resize',  action='store_true', default=False)
    parser.add_argument('--face',  action='store_true', default=False)
    args = parser.parse_args()
    modes = ['train', 'val']
    folder_root = '/home/afromero/ssd2/EmotionNet2018/data'

    option='|'
    if args.face:
        Faces = Face()
        face_root = folder_root.replace(os.path.basename(folder_root), 'faces') 
        option+='Faces|'
        if not os.path.isdir(face_root): os.makedirs(face_root)  
    if args.resize:
        if args.face:
            size_root = face_root.replace(os.path.basename(face_root), os.path.basename(face_root)+'_'+str(args.img_size))
        else:
            size_root = folder_root.replace(os.path.basename(folder_root), os.path.basename(folder_root)+'_'+str(args.img_size))
        option+='Resizing|'

        if not os.path.isdir(size_root): os.makedirs(size_root)  

    img_files = [sorted(glob.glob(folder_root+'/'+mode+'/*.jpg')) for mode in modes]

    for idx, mode in enumerate(modes):   
        for org_file in tqdm.tqdm(img_files[idx], total=len(img_files[idx]), \
                desc='%s Files [%s]'%(option,mode), ncols=80, leave=True):

            if args.face:
                face_file = org_file.replace(folder_root, face_root)   
                Faces.get_face_and_save(org_file, face_file)  
                if args.resize:
                    resize_file = org_file.replace(folder_root, size_root)  
                    get_resize(face_file, resize_file, args.img_size)

            elif args.resize:
                resize_file = org_file.replace(folder_root, size_root)    
                get_resize(org_file, resize_file, args.img_size)
