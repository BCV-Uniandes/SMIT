def config_GENERATOR(config, update_folder):

  if 'COLOR_JITTER' in config.GAN_options: update_folder(config, 'COLOR_JITTER')
  if 'BLUR' in config.GAN_options: update_folder(config, 'BLUR') 
  if 'GRAY' in config.GAN_options: update_folder(config, 'GRAY') 

  if 'RaGAN' in config.GAN_options: 
    update_folder(config, 'RaGAN')
    config.d_train_repeat = 1

  if 'L1_LOSS' in config.GAN_options: 
    update_folder(config, 'L1_LOSS') 
    update_folder(config, 'lambda_l1_10.0') 

  if 'TTUR' in config.GAN_options:
    update_folder(config, 'TTUR') 
    config.d_lr = 0.0004
    config.d_train_repeat = 1    

  if 'SpectralNorm' in config.GAN_options: 
    update_folder(config, 'SpectralNorm')
    
  if 'HINGE' in config.GAN_options: 
    update_folder(config, 'HINGE') 

  if 'Idt' in config.GAN_options: 
    update_folder(config, 'Idt')     

  if 'InterLabels' in config.GAN_options: 
    update_folder(config, 'InterLabels')    

  if 'content_loss' in config.GAN_options: 
    if not 'InterLabels' in config.GAN_options: 
      config.GAN_options.append('InterLabels')
      update_folder(config, 'InterLabels')    
    update_folder(config, 'content_loss_'+str(config.lambda_content))             

  if 'Attention' in config.GAN_options: 
    config.lambda_mask = 1
    config.lambda_mask_smooth = 0.0001
    update_folder(config, 'Attention')  

  if 'AdaIn' in config.GAN_options or 'Stochastic' in config.GAN_options:
    config.mlp_dim=256
    # config.style_dim=1#4

  if 'Stochastic' in config.GAN_options:
    # config.lambda_style = 1
    if not 'AdaIn' in config.GAN_options and not 'info_like' in config.GAN_options: config.GAN_options.append('AdaIn')    
    update_folder(config, 'Stochastic') 
    if config.style_dim==16:
      update_folder(config, 'style_'+str(config.lambda_style))
    else:
      update_folder(config, 'style_{}_dim_{}'.format(config.lambda_style, config.style_dim))
    

    if 'InterStyleLabels' in config.GAN_options: 
      if not 'InterLabels' in config.GAN_options: config.GAN_options.append('InterLabels')
      # if 'DRIT' in config.GAN_options: config.GAN_options.remove('DRIT')    
      update_folder(config, 'InterStyleLabels')      

    if 'DRIT' in config.GAN_options:   
      update_folder(config, 'DRIT')       
    if 'mono_style' in config.GAN_options:
      update_folder(config, 'mono_style')     
    if 'style_labels' in config.GAN_options:
      update_folder(config, 'style_labels') 
    if 'style_pseudo_random' in config.GAN_options:
      update_folder(config, 'style_pseudo_random')       
    if 'style_label_net' in config.GAN_options:
      if not 'style_labels' in config.GAN_options and 'style_pseudo_random' not in config.GAN_options: config.GAN_options.append('style_labels')
      update_folder(config, 'style_label_net')       

    if 'vae_like' in config.GAN_options:
      update_folder(config, 'vae_like') 
      if not 'kl_loss' in config.GAN_options: config.GAN_options.append('kl_loss')  

    if 'info_like' in config.GAN_options:
      update_folder(config, 'info_like') 
      if not 'kl_loss' in config.GAN_options: config.GAN_options.append('kl_loss')        

    if 'LOGVAR' in config.GAN_options:
      update_folder(config, 'LOGVAR')   

    if 'FC' in config.GAN_options: 
      update_folder(config, 'FC')     

    if 'kl_loss' in config.GAN_options: 
      # config.lambda_kl = 10
      update_folder(config, 'kl_loss_'+str(config.lambda_kl))
    if not 'Stochastic' in config.GAN_options: 
      update_folder(config, 'AdaIn')    

  if 'Split_Optim_all' in config.GAN_options: 
    update_folder(config, 'Split_Optim_all') 

  elif 'Split_Optim' in config.GAN_options: 
    update_folder(config, 'Split_Optim') 

  if 'Split_Data' in config.GAN_options: 
    update_folder(config, 'Split_Data')   

  if 'mse_style' in config.GAN_options: 
    update_folder(config, 'mse_style')          

  if config.batch_size==2:
    update_folder(config, 'bs_2') 

  if 'RaGAN' in config.GAN_options:
    config.batch_size *= 2

def update_folder(config, folder):
  import os
  config.log_path = os.path.join(config.log_path, folder)
  config.sample_path = os.path.join(config.sample_path, folder)
  config.model_save_path =os.path.join(config.model_save_path, folder)

def update_folder_generator(config, folder):
  import os
  config.Generator_path = os.path.join(config.Generator_path, folder)

def replace_folder_gan(config):
  import os
  replaced = 'snapshot'
  replace = os.path.join(replaced, config.mode_train, config.dataset_fake)
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
  if 'GOOGLE' in config.GAN_options or 'TEST' in config.GAN_options: config.mode='test'
  if 'VAL_SHOW' in config.GAN_options: config.mode='val'

  config.AUs = {'EMOTIONNET': [1, 2, 4, 5, 6, 9, 12, 17, 20, 25, 26, 43],
                'BP4D': [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24],
                'CELEBA': [1, 2, 4, 5, 6, 9, 12, 17, 20, 25, 26, 43],
                'MNIST': [],
                'DEMO': []} #CelebA AUs are for training framework
  config.AUs_Common=  [1, 2, 4, 6, 12, 17]

  replace_folder_gan(config)

  update_folder(config, os.path.join(config.mode_data, str(config.image_size), 'fold_'+config.fold))
  config.metadata_path = os.path.join(config.metadata_path, '{}', config.mode_data, 'fold_'+config.fold, )
  config.g_repeat_num = 6 if config.image_size <256 else 9 #int(math.log(config.image_size,2)-1)
  config.d_repeat_num = int(math.log(config.image_size,2)-1)
  config.mean=(0.5,0.5,0.5)
  config.std=(0.5,0.5,0.5)

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
