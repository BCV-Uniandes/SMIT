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
def _CLS_LOSS(output, target):
  """Compute binary or softmax cross entropy loss."""
  import torch.nn.functional as F
  return F.binary_cross_entropy_with_logits(output, target, size_average=False) / output.size(0)

#=======================================================================================#
#=======================================================================================#
def _GAN_LOSS(Disc, real_x, fake_x, label, opts, GEN=False):
  import torch
  import torch.nn.functional as F
  src_real, cls_real = Disc(real_x)
  src_fake, cls_fake = Disc(fake_x)

  # Relativistic GAN
  if 'RaGAN' in opts:
    if 'HINGE' in opts:
      loss_real = torch.mean(F.relu(1.0 - (src_real - torch.mean(src_fake))))
      loss_fake = torch.mean(F.relu(1.0 + (src_fake - torch.mean(src_real))))        
    else:
      #Wasserstein
      loss_real = torch.mean(-(src_real - torch.mean(src_fake)))
      loss_fake = torch.mean(src_fake - torch.mean(src_real))
    loss_src = (loss_real + loss_fake)/2 

  else:
    if 'HINGE' in opts:
      d_loss_real = torch.mean(F.relu(1-src_real))
      d_loss_fake = torch.mean(F.relu(1+src_fake))            
    else:
      #Wasserstein
      d_loss_real = -torch.mean(src_real)
      d_loss_fake = torch.mean(src_fake)   
    if GEN: loss_src = loss_real
    else: loss_src = loss_real + loss_fake  

  loss_cls = _CLS_LOSS(cls_real, label)

  return  loss_src, loss_cls

#=======================================================================================#
#=======================================================================================#
def _get_gradient_penalty(Disc, real_x, fake_x):
  import torch
  from utils import to_cuda, to_var
  alpha = to_cuda(torch.rand(real_x.size(0), 1, 1, 1).expand_as(real_x))
  interpolated = to_var((alpha * real_x.data + (1 - alpha) * fake_x.data), requires_grad = True)
  out, _ = Disc(interpolated)

  grad = torch.autograd.grad(outputs=out,
                 inputs=interpolated,
                 grad_outputs=to_cuda(torch.ones(out.size())),
                 retain_graph=True,
                 create_graph=True,
                 only_inputs=True)[0]

  grad = grad.view(grad.size(0), -1)
  grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
  d_loss_gp = torch.mean((grad_l2norm - 1)**2)   
  return d_loss_gp 