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

def update_folder(config, folder):
  import os
  config.log_path = os.path.join(config.log_path, folder)
  config.sample_path = os.path.join(config.sample_path, folder)
  config.model_save_path =os.path.join(config.model_save_path, folder)

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
  if config.GOOGLE or config.TEST: config.mode='test'
  if config.VAL_SHOW: config.mode='val'

  update_folder(config, os.path.join(config.mode_data, str(config.image_size), 'fold_'+config.fold))
  config.metadata_path = os.path.join(config.metadata_path, config.mode_data, 'fold_'+config.fold, )
  config.g_repeat_num = int(math.log(config.image_size,2)-1)
  config.d_repeat_num = int(math.log(config.image_size,2)-1)
  config.mean=(0.5,0.5,0.5)
  config.std=(0.5,0.5,0.5)

  if config.COLOR_JITTER: update_folder(config, 'COLOR_JITTER')

  if config.BLUR: update_folder(config, 'BLUR') 

  if config.GRAY: update_folder(config, 'GRAY') 

  if config.LSGAN: 
    update_folder(config, 'LSGAN') 
    config.d_train_repeat = 1

  if config.L1_LOSS: 
    update_folder(config, 'L1_LOSS') 
    if config.lambda_l1!=0.5: update_folder(config, 'lambda_l1_'+str(config.lambda_l1)) 

  if config.NO_TANH: update_folder(config, 'NO_TANH')

  if config.NEW_GEN: 
    update_folder(config, 'NEW_GEN') 
    config.batch_size=16

  if config.SAGAN: 
    update_folder(config, 'SAGAN')
    config.d_lr = 0.0004
    config.d_train_repeat = 1
    config.HINGE = True
    config.SpectralNorm = True
    config.batch_size=4

  if config.SpectralNorm: 
    update_folder(config, 'SpectralNorm')
    if not config.SAGAN: config.batch_size=8

  if config.HINGE: 
    update_folder(config, 'HINGE') 

  if config.DELETE:
    remove_folder(config)

  if config.pretrained_model is None:
    if config.LSTM:
      try:
        config.pretrained_model = sorted(glob.glob(os.path.join(config.model_save_path, '*_LSTM.pth')))[-1]
        config.pretrained_model = '_'.join(os.path.basename(config.pretrained_model).split('_')[:2])
      except:
        pass
    else:     
      try:
        # ipdb.set_trace()
        config.pretrained_model = sorted(glob.glob(os.path.join(config.model_save_path, '*_D.pth')))[-1]
        config.pretrained_model = '_'.join(os.path.basename(config.pretrained_model).split('_')[:-1])
      except:
        pass

  if config.test_model=='':
    config.test_model = config.pretrained_model

  return config
