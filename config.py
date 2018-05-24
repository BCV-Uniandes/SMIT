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
  if config.DENSENET: config.CLS=True
  if config.CLS: config.DENSENET=True

  if config.CLS:
     config.num_epochs = 29
     config.num_epochs_decay = 30

  if config.mean!='0.5' and not config.G_norm and not config.D_norm: raise TypeError('Normalization must be tied to any of the D or G networks')
  if config.G_norm or config.D_norm: assert config.mean!='0.5', "Select normalization"

  update_folder(config, os.path.join(config.mode_data, str(config.image_size), 'fold_'+config.fold))
  config.metadata_path = os.path.join(config.metadata_path, config.mode_data, 'fold_'+config.fold, )
  config.g_repeat_num = int(math.log(config.image_size,2)-1)
  config.d_repeat_num = int(math.log(config.image_size,2)-1)
  
  if config.mean!='0.5':
    update_folder(config, 'mean_{}'.format(config.mean))
    if config.G_norm: update_folder(config, 'G_norm')
    if config.D_norm: update_folder(config, 'D_norm') 

  elif config.mean=='0.5': 
    config.sample_step = 20000
    config.MEAN=config.mean
    config.mean=(0.5,0.5,0.5)
    config.std=(0.5,0.5,0.5)

  if config.mean=='data_mean' or config.std=='data_mean':
    # ipdb.set_trace()
    config.MEAN=config.mean
    mean_img = 'data/face_{}_mean.jpg'.format(config.mode_data)
    std_img = 'data/face_{}_std.jpg'.format(config.mode_data)
    print("Mean and Std from data: %s and %s"%(mean_img,std_img))
    mean = imageio.imread(mean_img).astype(np.float32).transpose(2,0,1)
    std = imageio.imread(std_img).astype(np.float32).transpose(2,0,1)

    mean = mean.mean(axis=(0,1))/255.
    std = std.std(axis=(0,1))/255.
    config.mean = (mean[0], mean[1], mean[2])

    config.std = (std[0], std[1], std[2])

    print("Mean {} and std {}".format(config.mean, config.std))

  elif config.mean=='data_full' or config.std=='data_full':
    # ipdb.set_trace()
    config.MEAN=config.mean
    config.mean=(0.0,0.0,0.0)
    config.std =(1.0,1.0,1.0)   

  elif config.mean=='data_image' or config.std=='data_image':
    config.MEAN=config.mean
    config.mean=(0.0,0.0,0.0)
    config.std =(1.0,1.0,1.0)

  if config.CelebA_CLS:
    config.CelebA = True
  else:
    config.CelebA = False

  if config.FAKE_CLS: update_folder(config, 'FAKE_CLS')
  if config.COLOR_JITTER: update_folder(config, 'COLOR_JITTER')
  if config.BLUR: update_folder(config, 'BLUR') 
  if config.GRAY: update_folder(config, 'GRAY') 
  if config.LSGAN: update_folder(config, 'LSGAN') 
  if config.L1_LOSS: update_folder(config, 'L1_LOSS') 
  if config.lambda_l1!=0.5: update_folder(config, 'lambda_l1_'+str(config.lambda_l1)) 
  if config.L2_LOSS: update_folder(config, 'L2_LOSS') 
  if config.CelebA_CLS: update_folder(config, 'CelebA_CLS')
  if config.JUST_REAL: update_folder(config, 'JUST_REAL')
  if config.DENSENET: update_folder(config, 'DENSENET')
  # if config.lambda_cls!=1: update_folder(config, 'lambda_cls_%d'%(config.lambda_cls))

  if config.CLS:
    config.pretrained_model_generator = sorted(glob.glob(os.path.join(config.model_save_path, '*_G.pth')))[-1]
    config.pretrained_model_discriminator = sorted(glob.glob(os.path.join(config.model_save_path, '*_D.pth')))[-1]

    config.log_path = config.log_path.replace('MultiLabelAU', 'MultiLabelAU_CLS')
    config.sample_path = config.sample_path.replace('MultiLabelAU', 'MultiLabelAU_CLS')
    config.model_save_path = config.model_save_path.replace('MultiLabelAU', 'MultiLabelAU_CLS')
    # config.result_path = config.result_path.replace('MultiLabelAU', 'MultiLabelAU_CLS')

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
    try:
      # ipdb.set_trace()
      config.test_model = sorted(glob.glob(os.path.join(config.model_save_path, '*_D.pth')))[-1]
      config.test_model = '_'.join(os.path.basename(config.test_model).split('_')[:-1])
    except:
      config.test_model = ''  

  return config