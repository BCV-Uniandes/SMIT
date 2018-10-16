def config_GENERATOR(config, update_folder):
  if config.dataset_fake=='CelebA':
    if config.ALL_ATTR==1:
      update_folder(config, 'ALL_ATTR')
      config.c_dim = 40
    elif config.ALL_ATTR==2:
      config.c_dim=37
      update_folder(config, 'ALL_ATTR_'+str(config.c_dim))      
    elif config.ALL_ATTR==3:
      config.num_epochs_decay = 30
      config.c_dim=25
      update_folder(config, 'ALL_ATTR_'+str(config.c_dim))
    elif config.ALL_ATTR==4:
      config.num_epochs_decay = 30
      config.c_dim=10
      update_folder(config, 'ALL_ATTR_'+str(config.c_dim))         
    elif config.ALL_ATTR==0:
      config.c_dim=5

  if config.dataset_fake=='AwA2':
    config.save_epoch = 6
    if config.ALL_ATTR==1:
      update_folder(config, 'ALL_ATTR')
      config.c_dim = 85
    elif config.ALL_ATTR==0:
      config.c_dim=6   

  if config.dataset_fake=='RafD':
    config.save_epoch = 20
    config.num_epochs = 40
    config.num_epochs_decay = 20
    if config.RafD_FRONTAL: config.save_epoch = 120
    if config.RafD_FRONTAL or config.RafD_EMOTIONS: config.c_dim=8
    else: config.c_dim=13

  if config.dataset_fake=='Birds':
    config.c_dim=5

  if config.dataset_fake=='painters_14':
    config.save_epoch = 20
    if config.ALL_ATTR==1:
      update_folder(config, 'ALL_ATTR')
      config.c_dim=14
    elif config.ALL_ATTR==0:
      config.c_dim=5   

  if config.dataset_fake=='Animals':
    if config.ALL_ATTR==1:
      update_folder(config, 'ALL_ATTR')
      config.save_epoch = 6
      config.c_dim=50
    elif config.ALL_ATTR==2:
      config.save_epoch = 10
      config.c_dim=17
      update_folder(config, 'ALL_ATTR_'+str(config.c_dim)) 
    elif config.ALL_ATTR==0:
      config.save_epoch = 30
      config.c_dim=8       

  if config.dataset_fake=='Image2Weather':
    if config.ALL_ATTR==1:
      update_folder(config, 'ALL_ATTR')
      # config.save_epoch = 6
      config.c_dim=5
    elif config.ALL_ATTR==0:
      config.save_epoch = 160
      config.c_dim=4           

  if config.dataset_fake=='Image2Season':
    config.save_epoch = 30
    if config.ALL_ATTR==1:
      # update_folder(config, 'ALL_ATTR')
      config.c_dim = 4
    elif config.ALL_ATTR==0:
      config.c_dim=4   

  if config.dataset_fake=='Image2Edges':
    # config.save_epoch = 30
    if config.ALL_ATTR==1:
      update_folder(config, 'ALL_ATTR')
      config.c_dim = 4
    elif config.ALL_ATTR==2:
      config.c_dim = 2
      update_folder(config, 'ALL_ATTR_Handbags') 
    elif config.ALL_ATTR==3:
      config.c_dim = 2 
      update_folder(config, 'ALL_ATTR_Shoes') 
    elif config.ALL_ATTR==0:
      config.c_dim=4         

  if config.dataset_fake=='WIDER':
    config.save_epoch = 30
    if config.ALL_ATTR==1:
      update_folder(config, 'ALL_ATTR')
      config.c_dim = 14
    elif config.ALL_ATTR==0:
      config.c_dim=5         

  if config.dataset_fake=='BAM':
    if config.ALL_ATTR==1:
      # update_folder(config, 'ALL_ATTR')
      config.c_dim = 20
    elif config.ALL_ATTR==0:
      config.c_dim=20   

  if config.dataset_fake=='EmotionNet':
    config.c_dim=12            
    config.num_epochs = 20 #The same iterations as celebA given the high number of images in the dataset
    config.num_epochs_decay = 6

  config.num_epochs *= config.save_epoch#500
  config.num_epochs_decay *= config.save_epoch#200    

  if config.MultiDis>0:
    update_folder(config, 'MultiDis_scale'+str(config.MultiDis))

  # if config.PerceptualLoss:
  if 'Perceptual' in config.GAN_options:
    update_folder(config, 'PerceptualLoss_{}_lambda_{}'.format(config.PerceptualLoss, config.lambda_perceptual))  

  if 'NoPerceptualIN' in config.GAN_options:
    update_folder(config, 'NoPerceptualIN')      

  if 'L1_Perceptual' in config.GAN_options: 
    update_folder(config, 'L1_Perceptual_'+str(config.lambda_l1perceptual)) 

  if 'COLOR_JITTER' in config.GAN_options: update_folder(config, 'COLOR_JITTER')
  if 'BLUR' in config.GAN_options: update_folder(config, 'BLUR') 
  if 'GRAY' in config.GAN_options: update_folder(config, 'GRAY') 

  if config.d_train_repeat!=5:
    update_folder(config, 'd_train_repeat_'+str(config.d_train_repeat))

  if config.lambda_cls!=4:
    update_folder(config, 'lambda_cls_'+str(config.lambda_cls))

  if 'RaGAN' in config.GAN_options: 
    update_folder(config, 'RaGAN')
    # config.d_train_repeat = 1

  if 'L1_LOSS' in config.GAN_options: 
    update_folder(config, 'L1_LOSS_'+str(config.lambda_l1)) 
    # if config.lambda_l1==1.0:
    #   update_folder(config, 'L1_LOSS') 
    #   update_folder(config, 'lambda_l1_10.0') 
    # else:
    #   update_folder(config, 'L1_LOSS_'+str(config.lambda_l1)) 

  if 'SpectralNorm' in config.GAN_options: 
    update_folder(config, 'SpectralNorm')
    
  if 'HINGE' in config.GAN_options: 
    update_folder(config, 'HINGE') 

  if 'InterLabels' in config.GAN_options: 
    update_folder(config, 'InterLabels')    

  if 'content_loss' in config.GAN_options: 
    # if not 'InterLabels' in config.GAN_options: 
    #   config.GAN_options.append('InterLabels')
    #   update_folder(config, 'InterLabels')    
    update_folder(config, 'content_loss_'+str(config.lambda_content))             

  if 'Attention' in config.GAN_options: 
    update_folder(config, 'Attention')  
  elif 'Attention2' in config.GAN_options: 
    update_folder(config, 'Attention2')      
  elif 'Attention3' in config.GAN_options: 
    update_folder(config, 'Attention3')          

  if 'AdaIn' in config.GAN_options and not 'Stochastic' in config.GAN_options:
    config.mlp_dim=256
    update_folder(config, 'AdaIn') 

  if 'Stochastic' in config.GAN_options:
    config.mlp_dim=256
    if not 'AdaIn' in config.GAN_options and not 'DRIT' in config.GAN_options and not 'DRITZ' in config.GAN_options:
      config.GAN_options.append('AdaIn')    
    update_folder(config, 'Stochastic') 
    update_folder(config, 'style_{}_dim_{}'.format(config.lambda_style, config.style_dim))   
    if config.style_dim==8 and not 'FC' in config.GAN_options: config.GAN_options.append('FC')


    if 'AdaIn2' in config.GAN_options:
      update_folder(config, 'AdaIn2') 

    if 'AdaIn3' in config.GAN_options:
      update_folder(config, 'AdaIn3')       

    if 'InterStyleLabels' in config.GAN_options: 
      if not 'InterLabels' in config.GAN_options: config.GAN_options.append('InterLabels')
      # if 'DRIT' in config.GAN_options: config.GAN_options.remove('DRIT')    
      update_folder(config, 'InterStyleLabels') 

    elif 'InterStyleConcatLabels' in config.GAN_options:
      update_folder(config, 'InterStyleConcatLabels')

    elif 'InterStyleMulLabels' in config.GAN_options:
      if not 'style_labels' in config.GAN_options: config.GAN_options.append('style_labels')
      update_folder(config, 'InterStyleMulLabels')      

    if 'DRITZ' in config.GAN_options: 
      if 'Split_Optim' in config.GAN_options: config.GAN_options.remove('Split_Optim')  
      update_folder(config, 'DRITZ')    

    elif 'DRIT' in config.GAN_options:   
      if 'Split_Optim' in config.GAN_options: config.GAN_options.remove('Split_Optim')
      update_folder(config, 'DRIT')    

    if 'style_labels' in config.GAN_options:
      config.style_label_debug = 1
      update_folder(config, 'style_labels') 

    if 'LOGVAR' in config.GAN_options:
      update_folder(config, 'LOGVAR')   

    if 'FC' in config.GAN_options: 
      update_folder(config, 'FC')     

    if 'kl_loss' in config.GAN_options: 
      update_folder(config, 'kl_loss_'+str(config.lambda_kl))
      if not 'Stochastic' in config.GAN_options: 
        update_folder(config, 'AdaIn')    

  if 'Split_Optim' in config.GAN_options: 
    update_folder(config, 'Split_Optim') 
  # if 'Stochastic' in config.GAN_options:
  #   update_folder(config, 'Split_Optim') 

  if 'mse_style' in config.GAN_options: 
    update_folder(config, 'mse_style') 

  if 'rec_style_gan' in config.GAN_options: 
    update_folder(config, 'rec_style_gan')   
    config.GAN_options.append('rec_style')

  elif 'rec_style' in config.GAN_options: 
    update_folder(config, 'rec_style')   
  elif 'ORG_REC' in config.GAN_options: 
    update_folder(config, 'ORG_REC')       

  if config.LOAD_SMIT: 
    update_folder(config, 'LOAD_SMIT') 

  if 'STYLE_DISC' in config.GAN_options: 
    update_folder(config, 'STYLE_DISC')    

  if 'GRAY_DISC' in config.GAN_options: 
    update_folder(config, 'GRAY_DISC') 

  if 'GRAY_STYLE' in config.GAN_options: 
    update_folder(config, 'GRAY_STYLE')     

  if 'CLS_L2' in config.GAN_options: 
    update_folder(config, 'CLS_L2') 

  if 'CLS_L1' in config.GAN_options: 
    update_folder(config, 'CLS_L1')     

  if 'AttentionStyle' in config.GAN_options: 
    update_folder(config, 'AttentionStyle') 

  if 'Identity' in config.GAN_options: 
    update_folder(config, 'Identity_'+str(config.lambda_idt))      

  if 'LayerNorm' in config.GAN_options: 
    update_folder(config, 'LayerNorm') 

  if config.dataset_smit: 
    update_folder(config, 'Finetuning_'+config.dataset_smit)         

  if 'RaGAN' in config.GAN_options:
    config.batch_size *= 2

  import torch
  if int(torch.__version__.split('.')[1])>3:
    update_folder(config, 'Pytorch_'+str(torch.__version__))

def update_folder(config, folder):
  import os
  config.log_path = os.path.join(config.log_path, folder)
  config.sample_path = os.path.join(config.sample_path, folder)
  config.model_save_path =os.path.join(config.model_save_path, folder)

def replace_folder_gan(config):
  import os
  replaced = 'snapshot'
  if config.RafD_FRONTAL: dataset = 'RafD_Frontal'
  elif config.RafD_EMOTIONS: dataset = 'RafD_EMOTIONS'
  else: dataset = config.dataset_fake
  replace = os.path.join(replaced, config.mode_train, dataset)
  config.log_path = config.log_path.replace(replaced, replace)
  config.sample_path = config.sample_path.replace(replaced, replace)
  config.model_save_path = config.model_save_path.replace(replaced, replace)

def remove_folder(config):
  import os
  logs = os.path.join(config.log_path, '*.bcv002')
  samples = os.path.join(config.sample_path, '*.jpg')
  samples_txt = os.path.join(config.sample_path, '*.txt')
  models = os.path.join(config.model_save_path, '*.pth')
  print("YOU ARE ABOUT TO REMOVE EVERYTHING IN:\n{}\n{}\n{}\n{}".format(logs, samples, samples_txt, models))
  raw_input("ARE YOU SURE?")
  os.system("rm {} {} {} {}".format(logs, samples, samples_txt, models))

def update_config(config):
  import os, glob, math, imageio, ipdb

  replace_folder_gan(config)

  update_folder(config, os.path.join(config.mode_data, str(config.image_size), 'fold_'+config.fold))
  config.metadata_path = os.path.join(config.metadata_path, '{}', config.mode_data, 'fold_'+config.fold, )
  config.g_repeat_num = 6 #if config.image_size <256  or ('Attention2' in config.GAN_options and 'InterStyleConcatLabels' in config.GAN_options) else 9 
  config.g_conv_dim = config.g_conv_dim if config.image_size<256 else config.g_conv_dim/2
  config.d_conv_dim = config.d_conv_dim if config.image_size<256 else config.d_conv_dim/2

  config.g_conv_dim = config.g_conv_dim if config.image_size<512 else config.g_conv_dim/2
  config.d_conv_dim = config.d_conv_dim if config.image_size<512 else config.d_conv_dim/2

  config.d_repeat_num = int(math.log(config.image_size,2)-1)

  config_GENERATOR(config, update_folder)

  if config.DELETE:
    remove_folder(config)

  config.dataset = config.dataset_fake

  if os.path.isdir('/home/afromero'):
    config.PLACE='BCV'
  else:
    config.PLACE='ETHZ'

  if config.pretrained_model is None:  
    try:
      config.pretrained_model = sorted(glob.glob(os.path.join(config.model_save_path, '*_D.pth')))[-1]
      config.pretrained_model = '_'.join(os.path.basename(config.pretrained_model).split('_')[:-1])
    except:
      pass

  return config
