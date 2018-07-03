#!/usr/bin/ipython
import glob
import os
gpus = sorted(glob.glob('logs/gpu*.txt'))
for gpu in gpus:
  line = open(gpu).readline().strip()
  print(os.path.basename(gpu)+'\t'+line)
