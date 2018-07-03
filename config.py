AUs = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]

AUs_name_en = ['Inner Brow Raiser\n', \
      'Outer Brow Raiser\n', \
      'Brow Lowerer\n', \
      'Cheek Raiser\n', \
      'Lid Tightener\n', \
      'Upper Lip Raiser\n', \
      'Lip Corner Puller\n', \
      'Dimpler\n', \
      'Lip Corner Depressor\n', \
      'Chin Raiser\n', \
      'Lip Tightener\n', \
      'Lip Pressor\n', \
      ]

AUs_name_es= ['Ceja Interior Levantada\n', \
      'Ceja Exterior Levantada\n', \
      'Cejas fruncidas\n', \
      'Mejillas levantadas\n', \
      'Parpados apretados\n', \
      'Labio Superior Levantado\n', \
      'Esquina Labial Estirada\n', \
      'Hoyuelo Facial\n', \
      'Esquina Labial hacia abajo\n', \
      'Barbilla levantada\n', \
      'Labios mordidos\n', \
      'Labios apretados\n', \
      ]

def config_GENERATOR(config, update_folder):
  if config.image_size<=128: config.batch_size=32
  if 'COLOR_JITTER' in config.GAN_options: update_folder(config, 'COLOR_JITTER')
  if 'BLUR' in config.GAN_options: update_folder(config, 'BLUR') 
  if 'GRAY' in config.GAN_options: update_folder(config, 'GRAY') 
  if 'LSGAN' in config.GAN_options: 
    update_folder(config, 'LSGAN') 
    config.d_train_repeat = 1

  if 'L1_LOSS' in config.GAN_options: 
    update_folder(config, 'L1_LOSS') 
    if config.lambda_l1!=0.5: update_folder(config, 'lambda_l1_'+str(config.lambda_l1)) 

  if 'SAGAN' in config.GAN_options: 
    update_folder(config, 'SAGAN')
    config.d_lr = 0.0004
    config.d_train_repeat = 1
    config.TTUR = True
    config.SpectralNorm = True
    config.batch_size=4
    if config.image_size==128: config.batch_size=50

  if 'TTUR' in config.GAN_options:
    update_folder(config, 'TTUR') 
    config.d_lr = 0.0004
    config.d_train_repeat = 1    

  if 'SpectralNorm' in config.GAN_options: 
    update_folder(config, 'SpectralNorm')
    if not 'SAGAN' in config.GAN_options and not 'GOOGLE' in config.GAN_options: 
      config.batch_size=8
      if config.image_size<=128: config.batch_size=32

  if 'HINGE' in config.GAN_options: 
    update_folder(config, 'HINGE') 

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

def replace_folder_cls(config):
  import os
  replaced = 'snapshot'
  replace = os.path.join(replaced, config.mode_train, '{}_to_{}'.format(config.dataset_fake, config.dataset_real))
  config.log_path = config.log_path.replace(replaced, replace)
  config.sample_path = config.sample_path.replace(replaced, replace)
  config.model_save_path = config.model_save_path.replace(replaced, replace)

def remove_folder(config):
  import os
  logs = os.path.join(config.log_path, '*.bcv002')
  samples = os.path.join(config.sample_path, '*.jpg')
  models = os.path.join(config.model_save_path, '*.pth')
  print("YOU ARE ABOUT TO REMOVE EVERYTHING IN:\n{}\n{}\n{}".format(logs, samples, models))
  raw_input("ARE YOU SURE?")
  os.system("rm {} {} {}".format(logs, samples, models))

def update_config(config):
  import os, glob, math, imageio
  if 'GOOGLE' in config.GAN_options or 'TEST' in config.GAN_options: config.mode='test'
  if 'VAL_SHOW' in config.GAN_options: config.mode='val'

  if config.dataset_fake=='EmotionNet':
    config.num_epochs = 19
    config.num_epochs_decay = 20

  if config.CLS_options!='':
    config.mode_train='CLS'

  if config.mode_train=='CLS': 
    replace_folder_cls(config)
  else:
    replace_folder_gan(config)

  update_folder(config, os.path.join(config.mode_data, str(config.image_size), 'fold_'+config.fold))
  config.metadata_path = os.path.join(config.metadata_path, '{}', config.mode_data, 'fold_'+config.fold, )
  config.g_repeat_num = int(math.log(config.image_size,2)-1)
  config.d_repeat_num = int(math.log(config.image_size,2)-1)
  config.mean=(0.5,0.5,0.5)
  config.std=(0.5,0.5,0.5)

  if config.mode_train=='CLS': 
    if config.Generator_path=='':
      config.Generator_path = config.model_save_path.replace(config.model_save_path.split('/')[3], config.model_save_path.split('/')[3].split('_')[0])
      config_GENERATOR(config, update_folder_generator)
      config.Generator_path = config.Generator_path.replace('CLS', 'GAN')
      # import ipdb;ipdb.set_trace()
      config.Generator_path = sorted(glob.glob(config.Generator_path+'/*G.pth'))[-1]
    # config.CLS_options.append('DENSENET')
    config.batch_size=32
    config.num_epochs = 5
    config.mean = 0.0
    config.std = 0.0
  else:
    config_GENERATOR(config, update_folder)

  if config.DELETE:
    remove_folder(config)

  if config.pretrained_model is None:  
    try:
      config.pretrained_model = sorted(glob.glob(os.path.join(config.model_save_path, '*_D.pth')))[-1]
      config.pretrained_model = '_'.join(os.path.basename(config.pretrained_model).split('_')[:-1])
    except:
      pass

  return config
