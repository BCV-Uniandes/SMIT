#=======================================================================================#
#=======================================================================================#
def create_dir(dir):
  import os
  if '.' in os.path.basename(dir):
    dir = os.path.dirname(dir)
  if not os.path.isdir(dir): os.makedirs(dir)

#=======================================================================================#
#=======================================================================================#
def denorm(x):
  out = (x + 1) / 2
  return out.clamp_(0, 1)

#=======================================================================================#
#=======================================================================================#
def get_aus(image_size, dataset, attr=None):
  import imageio, glob, torch
  import skimage.transform  
  import numpy as np
  import ipdb

  imread = lambda x: imageio.imread(x)
  resize = lambda x: skimage.transform.resize(x, (image_size, image_size))
  if dataset not in ['EmotionNet', 'BP4D']:
    from data.attr2img import external2img
    labels = ['Source']+attr.selected_attrs
    imgs = external2img(labels, img_size=image_size)
    imgs = [resize(np.array(img)).transpose(2,0,1) for img in imgs]
  else:
    imgs_file = sorted(glob.glob('data/{}/aus_flat/*g'.format(dataset)))
    imgs_file.pop(1) #Removing 'off'
    imgs = [resize(imread(line)).transpose(2,0,1) for line in imgs_file]
  imgs = torch.from_numpy(np.concatenate(imgs, axis=2).astype(np.float32)).unsqueeze(0)
  # from torchvision.utils import save_image; save_image(imgs, 'meh.jpg', nrow=1, padding=0)
  return imgs  

#=======================================================================================#
#=======================================================================================#
def get_loss_value(x):
  import torch
  if get_torch_version()>0.3:
    return x.item()
  else:
    return x.data[0]

#=======================================================================================#
#=======================================================================================#
def get_torch_version():
  import torch
  return float('.'.join(torch.__version__.split('.')[:2]))

#=======================================================================================#
#=======================================================================================#
def _horovod():
  try:
    import horovod.torch as hvd
  except:
    class hvd():
      def init(self):
        pass
      def size(self):
        return 1
      def rank(self):
        return 0
    hvd = hvd()    
  return hvd 

#=======================================================================================#
#=======================================================================================#
def imgShow(img):
  from torchvision.utils import save_image
  try:save_image(denorm(img).cpu(), 'dummy.jpg')
  except: save_image(denorm(img.data).cpu(), 'dummy.jpg')

#=======================================================================================#
#=======================================================================================#
def make_gif(imgs, path, im_size=256):
  import imageio, numpy as np, ipdb
  if 'jpg' in path: path = path.replace('jpg', 'gif')
  imgs = (imgs.cpu().numpy().transpose(0,2,3,1)*255).astype(np.uint8)
  # ipdb.set_trace()
  target_size = (im_size, im_size, imgs.shape[-1])
  img_list = []
  for x in range(imgs.shape[2]//im_size):
    for bs in range(imgs.shape[0]):
      if x==0 and bs>1: continue #Only save one image of the originals
      if x==1: continue #Do not save any of the 'off' label
      img_short = imgs[bs,:,im_size*x:im_size*(x+1)]
      assert img_short.shape==target_size
      img_list.append(img_short)
  imageio.mimsave(path, img_list, duration=0.8)

  writer = imageio.get_writer(path.replace('gif','mp4'), fps=3)
  for im in img_list:
      writer.append_data(im)
  writer.close()

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
def replace_weights(target, source, list):
  for l in list:
    target[l] = source[l] 

#=======================================================================================#
#=======================================================================================#
def slerp(val, low, high):
  """
  original: Animating Rotation with Quaternion Curves, Ken Shoemake
  https://arxiv.org/abs/1609.04468
  Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
  """
  import numpy as np
  omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
  so = np.sin(omega)
  if so == 0:
    return (1.0-val) * low + val * high # L'Hopital's rule/LERP  
  return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high  

#=======================================================================================#
#=======================================================================================#
def send_mail(body="bcv002", attach=[], subject='Message from bcv002', to='rv.andres10@uniandes.edu.co'):
  import os,ipdb,time
  content_type = {'jpg':'image/jpeg', 'gif':'image/gif', 'mp4':'video/mp4'}
  if len(attach): #Must be a list with the files
    enclosed = []
    for line in attach:
      format = line.split('.')[-1]
      enclosed.append('--content-type={} --attach {}'.format(content_type[format], line))
    enclosed = ' '.join(enclosed)
  else:
    enclosed = ''
  mail = 'echo "{}" | mail -s "{}" {} {}'.format(body, subject, enclosed, to)
  # print(mail)
  os.system(mail)

#=======================================================================================#
#=======================================================================================#
def target_debug_list(size, dim, config=None):
  import torch, ipdb
  target_c= torch.zeros(size, dim)
  target_c_list = []
  for j in range(dim+1):
    target_c[:]=0 
    # if config.dataset_fake in ['RafD', 'painters_14', 'Animals', 'Image2Weather', 'Image2Season'] and j==0: continue
    # if j==0: continue
    if j>0: target_c[:,j-1]=1 
    if not config.RafD_FRONTAL:
      if config.dataset_fake=='RafD' and j==0: target_c[:,2] = 1
      if config.dataset_fake=='RafD' and j<=5: target_c[:,10] = 1
      if config.dataset_fake=='RafD' and j>=6: target_c[:,2] = 1
    else:
      if config.dataset_fake=='RafD' and j==0: target_c[:,0] = 1

    if config.dataset_fake=='BAM' and j<=10: 
      target_c[:,10] = 1; target_c[:,13] = 1
    elif config.dataset_fake=='BAM' and j<=14: 
      target_c[:,0] = 1; target_c[:,13] = 1
    elif config.dataset_fake=='BAM' and j<=20: 
      target_c[:,0] = 1; target_c[:,10] = 1
    # ipdb.set_trace()     
# 9, 4, 7
# 'content_bicycle', 'content_bird', 'content_building', 'content_cars', 'content_cat', 'content_dog', 'content_flower', 'content_people', 'content_tree', 
# 'emotion_gloomy', 'emotion_happy', 'emotion_peaceful', 'emotion_scary', 
# 'media_3d_graphics', 'media_comic', 'media_graphite', 'media_oilpaint', 'media_pen_ink', 'media_vectorart', 'media_watercolor'      

    # ipdb.set_trace()
    target_c_list.append(to_var(target_c, volatile=True))        
  return target_c_list

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
def to_cpu(x):
  return x.cpu() if x.is_cuda else x

#=======================================================================================#
#=======================================================================================#
def to_cuda(x):
  import torch
  import torch.nn as nn
  if get_torch_version()>0.3:
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
def to_data(x, cpu=False):
  import torch
  if get_torch_version()>0.3:
    x = x.data
  else:
    from torch.autograd import Variable
    if isinstance(x, Variable): x = x.data
  if cpu: x = to_cpu(x)
  return x

#=======================================================================================#
#=======================================================================================#
def to_parallel(main, input, list_gpu):
  import torch
  import torch.nn as nn
  import ipdb
  if len(list_gpu)>1 and input.is_cuda:
    if get_torch_version()>0.3:
      return nn.parallel.data_parallel(main, input,  device_ids = list_gpu)
    else:  
      return nn.parallel.data_parallel(main, input,  device_ids = list_gpu)
      # ipdb.set_trace()
  else:
    return main(input)

#=======================================================================================#
#=======================================================================================#
def to_var(x, volatile=False, requires_grad=False, no_cuda=False):
  import torch
  if not no_cuda: x = to_cuda(x)
  if get_torch_version()>0.3:
    if requires_grad:
      return x.requires_grad_(True)
    else:
      return x

  else:
    from torch.autograd import Variable      
    if isinstance(x, Variable): return x
    return Variable(x, volatile=volatile, requires_grad=requires_grad)  

#=======================================================================================#
#=======================================================================================#
def vgg_preprocess(batch, meta):
  import torch, ipdb
  if batch.size(-1)==128:
    batch = batch.repeat(1,1,2,2)

  if batch.size(-1)>=256:
    center = batch.size(-1)/2
    vgg_size = meta['imSize']
    batch = batch[:,:,center-vgg_size[0]/2:center+vgg_size[0]/2,center-vgg_size[1]/2:center+vgg_size[1]/2]

  tensortype = type(batch.data)
  if meta['name']=='ImageNet' or meta['name']=='EmoNet':
    batch = (batch + 1) * 0.5 # [-1, 1] -> [0, 1]

  elif meta['name']=='DeepFace' or meta['name']=='Style':
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
  else:
    raise TypeError('You must preprocess data.')

  mean = tensortype(batch.data.size())
  mean[:, 0, :, :] = meta['mean'][0] #103.939
  mean[:, 1, :, :] = meta['mean'][1] #116.779
  mean[:, 2, :, :] = meta['mean'][2] #123.680

  std = tensortype(batch.data.size())
  std[:, 0, :, :] = meta['std'][0] 
  std[:, 1, :, :] = meta['std'][1] 
  std[:, 2, :, :] = meta['std'][2] 

  batch = batch.sub(to_var(mean)) # subtract mean
  batch = batch.div(to_var(std))  # divide std
  return batch    