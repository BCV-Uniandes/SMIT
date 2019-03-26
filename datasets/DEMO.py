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
                 Detect_Face=False,
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
        self.Detect_Face = Detect_Face
        self.len = len(self.lines)

    def __getitem__(self, index):
        if self.Detect_Face:
            image, success = self.face.get_face_from_file(
                self.img_path, margin=5.)
            if not success:
                image = Image.open(self.lines[index]).convert('RGB')
            else:
                image = Image.fromarray(image).convert('RGB')
        else:
            image = Image.open(self.lines[index]).convert('RGB')
        return self.transform(image)

    def __len__(self):
        return self.len
