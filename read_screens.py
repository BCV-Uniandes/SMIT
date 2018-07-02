#!/usr/bin/ipython
import glob
gpus = sorted(glob.glob('logs/gpu*.txt'))
for gpu in gpus:
  line = open(gpu).readline().strip()
  print(line)
