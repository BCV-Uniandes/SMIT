#=======================================================================================#
#=======================================================================================#
def denorm(x):
  out = (x + 1) / 2
  return out.clamp_(0, 1)

#=======================================================================================#
#=======================================================================================#
def get_aus(image_size, dataset):
  import imageio, glob, torch
  import skimage.transform  
  import numpy as np
  resize = lambda x: skimage.transform.resize(imageio.imread(line), (image_size, image_size))
  imgs = [resize(line).transpose(2,0,1) for line in sorted(glob.glob('data/{}/aus_flat/*.jpeg'.format(dataset)))]
  # ipdb.set_trace()
  if self.config.dataset_fake=='MNIST': 
    imgs = [np.expand_dims(img.mean(axis=0),0) for img in imgs]    
  imgs = torch.from_numpy(np.concatenate(imgs, axis=2).astype(np.float32)).unsqueeze(0)
  return imgs  

#=======================================================================================#
#=======================================================================================#
def get_loss_value(x):
  import torch
  if int(torch.__version__.split('.')[1])>3:
    return x.item()
  else:
    return x.data[0]

#=======================================================================================#
#=======================================================================================#
def imgShow(img):
  from torchvision.utils import save_image
  try:save_image(denorm(img).cpu(), 'dummy.jpg')
  except: save_image(denorm(img.data).cpu(), 'dummy.jpg')
  #os.system('eog dummy.jpg')  
  #os.remove('dummy.jpg')    

#=======================================================================================#
#=======================================================================================#
def one_hot(labels, dim):
  """Convert label indices to one-hot vector"""
  import torch
  import numpy as np
  batch_size = labels.size(0)
  out = torch.zeros(batch_size, dim)
  out[np.arange(batch_size), labels.long()] = 1
  return out

#=======================================================================================#
#=======================================================================================#
def PRINT(file, str):  
  print >> file, str
  file.flush()
  print(str)  

#=======================================================================================#
#=======================================================================================#
def pdf2png(filename):
  from wand.image import Image
  from wand.color import Color
  with Image(filename="{}.pdf".format(filename), resolution=500) as img:
    with Image(width=img.width, height=img.height, background=Color("white")) as bg:
      bg.composite(img,0,0)
      bg.save(filename="{}.png".format(filename))
  os.remove('{}.pdf'.format(filename))

#=======================================================================================#
#=======================================================================================#
def TimeNow():
  import datetime, pytz
  return str(datetime.datetime.now(pytz.timezone('Europe/Amsterdam'))).split('.')[0]

#=======================================================================================#
#=======================================================================================#
def TimeNow_str():
  import re
  return re.sub('\D','_', TimeNow())  

#=======================================================================================#
#=======================================================================================#
def to_cuda(x):
  import torch
  import torch.nn as nn
  if int(torch.__version__.split('.')[1])>3:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(x, nn.Module):
      x.to(device)
    else:
      return x.to(device)
  else:
    if torch.cuda.is_available():
      if isinstance(x, nn.Module):
        x.cuda()
      else:
        return x.cuda()
    else:
      return x

#=======================================================================================#
#=======================================================================================#
def to_var(x, volatile=False, requires_grad=False):
  import torch
  if int(torch.__version__.split('.')[1])>3:
    return to_cuda(x)
  else:
    from torch.autograd import Variable      
    return Variable(to_cuda(x), volatile=volatile, requires_grad=requires_grad)  