import os
from torch.utils.data import Dataset
from PIL import Image
import glob
from generate_data import Face

# ==================================================================#
# == DEMO
# ==================================================================#


class DEMO(Dataset):
    def __init__(self,
                 image_size,
                 img_path,
                 transform,
                 mode,
                 shuffling=False,
                 many_faces=False,
                 **kwargs):
        self.img_path = img_path
        self.transform = transform

        if os.path.isdir(img_path):
            self.lines = sorted(
                glob.glob(os.path.join(img_path, '*.jpg')) +
                glob.glob(os.path.join(img_path, '*.png')))
        else:
            self.lines = [self.img_path]

        self.face = Face()
        self.many_faces = many_faces
        if many_faces:
            self.lines = self.face.get_all_faces_from_file(
                self.img_path, margin=3.)

        self.len = len(self.lines)

    def __getitem__(self, index):
        if not self.many_faces:
            success = False
            if not success:
                image = Image.open(self.lines[index]).convert('RGB')
            else:
                image = Image.fromarray(image).convert('RGB')

        else:
            import imageio
            image = imageio.imread(self.img_path)
            bbox = self.lines[index]
            image = image[max(bbox[1], 0):min(bbox[3], image.shape[0]),
                          max(bbox[0], 0):min(bbox[2], image.shape[1])]
            image = Image.fromarray(image)
        return self.transform(image)

    def __len__(self):
        return self.len
