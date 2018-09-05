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
  import ipdb
  resize = lambda x: skimage.transform.resize(imageio.imread(line), (image_size, image_size))
  imgs = [resize(line).transpose(2,0,1) for line in sorted(glob.glob('data/{}/aus_flat/*g'.format(dataset)))]
  # ipdb.set_trace()   
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

#=======================================================================================#
#=======================================================================================#
def make_gif(imgs, path):
  import imageio, numpy as np
  if 'jpg' in path: path = path.replace('jpg', 'gif')
  imgs = (imgs[1:].cpu().numpy().transpose(0,2,3,1)*255).astype(np.uint8)
  size = imgs.shape[1]
  target_size = (imgs.shape[1], imgs.shape[1], imgs.shape[-1])
  img_list = []
  for bs in range(imgs.shape[0]):
    for x in range(imgs.shape[2]//imgs.shape[1]):
      img_short = imgs[bs,:,size*x:size*(x+1)]
      assert img_short.shape==target_size
      img_list.append(img_short)
  imageio.mimsave(path, img_list, duration=0.3)

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
def to_var(x, volatile=False, requires_grad=False, no_cuda=False):
  import torch
  if not no_cuda: x = to_cuda(x)
  if int(torch.__version__.split('.')[1])>3:
    return x
  else:
    from torch.autograd import Variable      
    return Variable(x, volatile=volatile, requires_grad=requires_grad)  

#=======================================================================================#
#=======================================================================================#
def vgg_preprocess(batch, meta):
  import torch
  tensortype = type(batch.data)
  if meta['name']=='DeepFace':
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
  batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
  # batch = resize(batch, )
  mean = tensortype(batch.data.size())
  mean[:, 0, :, :] = meta['mean'][0] #103.939
  mean[:, 1, :, :] = meta['mean'][1] #116.779
  mean[:, 2, :, :] = meta['mean'][2] #123.680
  batch = batch.sub(to_var(mean)) # subtract mean
  return batch    