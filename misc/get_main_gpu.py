#!/usr/bin/python
import os, argparse
__GPU__ = [0,1,2,3]
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--test',  action='store_true', default=False)
  parser.add_argument('--folder',  action='store_true', default=False)
  parser.add_argument('--last_name',  action='store_true', default=False)
  config = parser.parse_args()
  for gpu in __GPU__:
    # print("GPU: "+str(gpu))
    temp = '.gpu{}.txt'.format(gpu)
    look = './main.py'
    os.system('ps -ef | grep "GPU={0}" > {1}'.format(gpu, temp))
    lines = [look + line.strip().split(look)[-1] for line in open(temp).readlines()]
    lines = [line for line in lines if 'pts' not in line]
    lines = list(set(lines))
    lines = [line.replace('--DELETE','') for line in lines]
    if config.test: 
      lines = [line + ' --mode=test --dataset_real=EmotionNet' for line in lines]
    if config.folder: 
      lines = [line + ' --FOLDER' for line in lines]      
    for line in lines:
      if config.folder:
        os.system(line)
        print('')
      else:
        print(line)
    print("------")
    os.remove(temp)
