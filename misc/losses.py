#=======================================================================================#
#=======================================================================================#
def _compute_kl(style):
  import torch
  # def _compute_kl(self, mu, sd):
  # mu_2 = torch.pow(mu, 2)
  # sd_2 = torch.pow(sd, 2)
  # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
  # return encoding_loss
  if len(style)==1:
    mu = style[0]
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)      
  else:
    mu = style[1]
    var = style[2]
    kl_element = mu.pow(2).add_(var.exp()).mul_(-1).add_(1).add_(var)
    encoding_loss = torch.sum(kl_element).mul_(-0.5)

  return encoding_loss

#=======================================================================================#
#=======================================================================================#
def _compute_loss_smooth(mat):
  import torch
  return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
        torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))    

#=======================================================================================#
#=======================================================================================#
def _compute_vgg_loss(vgg, img, target):
  import torch, ipdb
  import torch.nn as nn
  from utils import vgg_preprocess, to_cuda
  def instancenorm(_img):
    _instancenorm = nn.InstanceNorm2d(512, affine=False)
    to_cuda(_instancenorm)
    return _instancenorm(_img)
    # return _img
  img_vgg = vgg_preprocess(img, vgg.meta)
  target_vgg = vgg_preprocess(target, vgg.meta)
  img_fea = vgg(img_vgg)
  target_fea = vgg(target_vgg)
  return torch.mean((instancenorm(img_fea) - instancenorm(target_fea)) ** 2)

#=======================================================================================#
#=======================================================================================#
def _CLS_LOSS(output, target):
  import torch.nn.functional as F
  return F.binary_cross_entropy_with_logits(output, target, size_average=False) / output.size(0)

#=======================================================================================#
#=======================================================================================#
def _GAN_LOSS(Disc, real_x, fake_x, label, opts, is_fake=False):
  import torch
  import torch.nn.functional as F
  src_real, cls_real, style_real = Disc(real_x)
  src_fake, cls_fake, style_fake = Disc(fake_x)

  loss_src = 0; loss_cls = 0
  for i in range(len(src_real)):
    # Relativistic GAN
    if 'RaGAN' in opts:
      if 'HINGE' in opts:
        loss_real = torch.mean(F.relu(1.0 - (src_real[i] - torch.mean(src_fake[i]))))
        loss_fake = torch.mean(F.relu(1.0 + (src_fake[i] - torch.mean(src_real[i]))))  
      else:
        #Wasserstein
        loss_real = torch.mean(-(src_real[i] - torch.mean(src_fake[i])))
        loss_fake = torch.mean(src_fake[i] - torch.mean(src_real[i]))
      loss_src += (loss_real + loss_fake)/2 

    else:
      if 'HINGE' in opts:
        loss_real = torch.mean(F.relu(1-src_real[i]))
        loss_fake = torch.mean(F.relu(1+src_fake[i]))            
      else:
        #Wasserstein
        loss_real = -torch.mean(src_real[i])
        loss_fake = torch.mean(src_fake[i])   
      if is_fake: loss_src += loss_real
      else: loss_src += loss_real + loss_fake  

    loss_cls += _CLS_LOSS(cls_real[i], label)

  if is_fake and 'StyleDisc' in opts:
    return loss_src, loss_cls, style_real[0], style_fake[0]
  else:
    return  loss_src, loss_cls

#=======================================================================================#
#=======================================================================================#
def _get_gradient_penalty(Disc, real_x, fake_x):
  import torch
  from utils import to_cuda, to_var
  alpha = to_cuda(torch.rand(real_x.size(0), 1, 1, 1).expand_as(real_x))
  interpolated = to_var((alpha * real_x + (1 - alpha) * fake_x), requires_grad = True)
  out, _, _ = Disc(interpolated)
  d_loss_gp = 0
  for idx in range(len(out)):
    grad = torch.autograd.grad(outputs=out[idx],
                   inputs=interpolated,
                   grad_outputs=to_cuda(torch.ones(out[idx].size())),
                   retain_graph=True,
                   create_graph=True,
                   only_inputs=True)[0]

    grad = grad.view(grad.size(0), -1)
    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
    d_loss_gp += torch.mean((grad_l2norm - 1)**2)   
  return d_loss_gp 
