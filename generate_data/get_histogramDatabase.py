# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 18:59:15 2015

@author: afromero
"""

import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import pylab as pyl
import glob
import ipdb
AUs = np.array([1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24])

def save_histogram(AUs_Frecuency, len_, mode, file_name):
    plt.figure(figsize=(13.0, 8.0))       
    ax = plt.axes()
    #ipdb.set_trace()
    position_hist = np.arange(len(AUs_Frecuency))
    ax.set_xticks(position_hist + 1.0/2)
    ax.set_xticklabels([0]+list(AUs), fontsize=10)
    ax.set_xlabel('AUs')
    ax.set_ylabel('Frequency')
    ax.set_title('Frequency of AUs for '+mode+' - '+str(len_)+' images')
    #ax.set_ylim([0, 60000])
    plt.bar(position_hist, AUs_Frecuency.values(), 1, color='b')
    if not os.path.isdir(os.path.dirname(file_name)): os.system('mkdir -p '+os.path.dirname(file_name))
    pyl.savefig(file_name, dpi=100)

if __name__ == '__main__':
    files_txt = glob.glob('../data/MultiLabelAU/normal/fold_*/Test.txt')
    AU = {au:0 for au in AUs}
    AU[0]=0
    total=0
    for file_txt in files_txt:
        lines = [line.strip().split(' ')[1:] for line in open(file_txt).readlines()]
	lines = [list(set(list(np.array(map(int, line))*AUs))) for line in lines]
        lines = [line if len(line)==1 and line[0]==0 else line[1:] for line in lines]
	for line in lines:
	    total+=1
	    for au in line:
                AU[au]+=1
    save_histogram(AU, total, 'BP4D', os.path.join('Histograms/BP4D.png'))
