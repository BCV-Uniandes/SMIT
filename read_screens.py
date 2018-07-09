#!/usr/bin/ipython
import glob
import os
gpus = sorted(glob.glob('logs/gpu*.txt'))
for gpu in gpus:
  line0 = open(gpu).readline().strip()
  line100 = [line.strip() for line in open(gpu).readlines()[-3:]]
  print('{}\t{}\n{}\n\n'.format(os.path.basename(gpu),line0,'\n'.join(line100)))
