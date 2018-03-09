import numpy as np
import theano
import theano.tensor as T
import lasagne


import skimage.transform
import sklearn.cross_validation
import pickle
import os
import re

##build the vgg model

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer,DropoutLayer,ROI_SliceLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax,sigmoid
from lasagne.utils import floatX
from lasagne.layers import SliceLayer, concat,BatchNormLayer,ElemwiseSumLayer,ElemwiseMergeLayer,ReshapeLayer
from lasagne.layers import LocalResponseNormalization2DLayer,BatchNormLayer, ROI_CropLayer,ROI_GotLayer,Upscale2DLayer
from lasagne.objectives import squared_error
from lasagne.layers import LSTMLayer

import get_bp4d_2dfeat
import get_atten_map
import random

IM_SIZE=224
BATCH_SIZE = 50


def get_f1_acc(outputs,y_labels):

    outputs_i=outputs+0.5
    outputs_i=outputs_i.astype('int32')
    y_ilab=y_labels.astype('int32')
    gd_num=T.sum(y_ilab,axis=0)
    pr_num=T.sum(outputs_i,axis=0)
    # pr_rtm=T.eq(outputs_i,y_ilab)
    # pr_rt=T.sum(pr_rtm,axis=0)
    
    sum_ones=y_ilab+outputs_i
    pr_rtm=sum_ones/2

    # pr_rtm=T.eq(outputs_i,y_ilab)
    pr_rt=T.sum(pr_rtm,axis=0)

    #prevent nan to destroy the f1
    pr_rt=pr_rt.astype('float32')
    gd_num=gd_num.astype('float32')
    pr_num=pr_num.astype('float32')

    acc=pr_rt/outputs.shape[0]

    zero_scale=T.zeros_like(T.min(pr_rt))
    if T.eq(zero_scale,T.min(gd_num)):
        gd_num+=1
    if T.eq(zero_scale,T.min(pr_num)):
        pr_num+=1
    if T.eq(zero_scale,T.min(pr_rt)):
        pr_rt+=0.01

    recall=pr_rt/gd_num
    precision=pr_rt/pr_num
    f1=2*recall*precision/(recall+precision)
    # return T.min(pr_rt)
    return acc,f1
def get_f1_acc_test(outputs,y_labels):
    # outputs+=0.5
    # print outputs.shape
    # outputs=outputs.astype('int8')
    # y_labels=y_labels.astype('int8')
    acc=np.zeros((12,))
    f1=np.zeros((12,))
    # add_rate=[0.65,0.7,0.65,0.5,0.5,0.5,0.5,0.5,0.85,0.7,0.8,0.7]
    add_rate=[0.5]*12
    for i in range(12):
        outputs_i=outputs[:,i]+add_rate[i]
    # print outputs.shape
        outputs_i=outputs_i.astype('int8')
        y_labels=y_labels.astype('int8')
        cnt=0
        acc[i]=sum(outputs_i==y_labels[:,i])/float(outputs.shape[0])
        gd_num=0
        pr_num=0
        pr_rt=0
        for j in range(outputs.shape[0]):
            if y_labels[j][i]==1:
                gd_num+=1
            if outputs_i[j]==1:
                pr_num+=1
            if y_labels[j][i]==1 and outputs_i[j]==1:
                pr_rt+=1
        if gd_num==0 or pr_num==0:
            continue
        recall=float(pr_rt)/gd_num
        precision=float(pr_rt)/pr_num
        # print 'AU',i,':',pr_rt,gd_num,pr_num
        f1[i]=2*recall*precision/(recall+precision)
    return acc,f1
def multi_label_ACE(outputs,y_labels):
    data_shape=outputs.shape
    loss_buff=0
    # num=T.iscalar(data_shape[0]) #theano int to get value from tensor
    # for i in range(int(num)):
    #     for j in range(12):
    #         y_exp=outputs[i,j]
    #         y_tru=y_labels[i,0,0,j]
    #         if y_tru==0:
    #             loss_ij=math.log(1-outputs[i,j])
    #             loss_buff-=loss_ij
    #         if y_tru>0:
    #             loss_ij=math.log(outputs[i,j])
    #             loss_buff-=loss_ij
    
    # wts=[ 0.24331649,  0.18382575,  0.23082499,  0.44545567,  0.52901483,  0.58482504, \
    # 0.57321465,  0.43411294,  0.15502839,  0.36377019,  0.19050646,  0.16083916]
    # for i in [3,4,5,6,7,9]:

    for i in range(12):
        target=y_labels[:,i]
        output=outputs[:,i]
        loss_au=T.sum(-(target * T.log((output+0.05)/1.05) + (1.0 - target) * T.log((1.05 - output)/1.05)))
        loss_buff+=loss_au
    return loss_buff/(12*BATCH_SIZE)






LS_X_sym=T.tensor3()
LS_y_sym=T.matrix()



def build_tempral_model():
    net={}
    net['input']=InputLayer((None,24,2048))
    net['lstm1']=LSTMLayer(net['input'],256)
    net['fc']=DenseLayer(net['lstm1'],num_units=12,nonlinearity=sigmoid)

    return net

LS_y_lb=LS_y_sym.reshape((LS_y_sym.shape[0],-1))

lstm_net=build_tempral_model()

# with np.load('data/LSTM2p_fusion_model.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]


# lasagne.layers.set_all_param_values(lstm_net['fc'], param_values)
print 'successfully loaded model.'

lstm_out = lasagne.layers.get_output(lstm_net['fc'], LS_X_sym)

loss=multi_label_ACE(lstm_out,LS_y_lb)

acc_scr,f1_score=get_f1_acc(lstm_out,LS_y_lb)


params = lasagne.layers.get_all_params(lstm_net['fc'], trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.001, momentum=0.9)

#model_BP4D_ROI_croping_withUpscl_iter1000_results_fold3.npz




train_LSfn = theano.function([LS_X_sym, LS_y_sym], loss, updates=updates)
print  'compile LSTM train'
val_LSfn = theano.function([LS_X_sym, LS_y_sym], loss)
print 'complie LSTM test'
f1_LSfn=theano.function([LS_X_sym, LS_y_sym],f1_score)
print 'compile LSTM F1'
pred_LSfn=theano.function([LS_X_sym],lstm_out)


import os
fp=open('../DATA/BP4D_SAD_tr.txt')
fp2=open('../DATA/BP4D_SAD_ts.txt')
line1=fp.readlines()
line2=fp2.readlines()
lines=line1+line2
lines.sort()
dic={}
for i,f in enumerate(lines):
    key=f.split('.')[0]
    dic[key]=i
    # print key,i
st_frame={}
for f in lines:
    key=f.split('.')[0]
    subj_id=key[:7]
    frame_num=int(key[8:])
    if subj_id not in st_frame:
        st_frame[subj_id]=frame_num;
    else:
        if frame_num<st_frame[subj_id]:
            st_frame[subj_id]=frame_num

print 'get dictionary'

def train_batch():
    trdata,trlb=LSTM_input(imglist)
    # trdata=trdata-MEAN_IMAGE
    return train_LSfn(trdata,trlb)

def test_batch():
    tsdata,tslb=LSTM_input(ixx)
    # trdata=trdata-MEAN_IMAGE
    loss=val_LSfn(tsdata,tslb)
    # batch_error=error_fn(tsdata,tslb)
    return loss

def batches(iterable, N):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk)==N:
            rst=chunk
            chunk=[]
            yield rst
    if chunk:
        yield chunk

def gen_ind(ind):
    num_id_front=int(ind.split('_')[-1]) #last number
    idkey=ind[:7] # M01_T03
    start_num=st_frame[idkey] #first frame lst number
    numrange=range(start_num,num_id_front) #all frames before boss frame
    random.shuffle(numrange)
    pickednums=[num_id_front]*30
    if len(numrange)<30:
        pickednums[:len(numrange)]=numrange
    else:
        pickednums=numrange[:30]

    # print len(pickednums)
    # pickednums=sorted(pickednums)
    # pickednums+=[num_id_front] #the 24 frame number

    new_ind=[lines[dic[ind]]]*24 #buff to fill
    
    #2 cases, if there's 96 valid frames, do this otherwise repeat the last one!
    
    num_bit=len(ind.split('_')[-1]) #3 or 4
    rp_str=str(start_num).zfill(num_bit)    
    st_frm=ind[:8]+rp_str
    # try:
    cnt=0
    for i in range(30):
        num_bit=len(ind.split('_')[-1])# 3 or 4
        rp_str=str(pickednums[i]).zfill(num_bit) # 0025? 
        key_frm=ind[:8]+rp_str  #full name for npy match
        if key_frm in dic:
            # print key_frm
            all_list_ind=dic[key_frm] #find the position
        # else:
            
        #     all_list_ind=num_id_front #else last one
            new_ind[cnt]=lines[all_list_ind]
            cnt+=1
            if cnt==23:
                break
    # except Exception as e:
    #     print 'error loading', ind
    new_ind.sort()       
    return new_ind

patt=re.compile('\d+')
def LSTM_input(fls,data_size=BATCH_SIZE):
    random.shuffle(fls)
    fls=fls[:data_size]
    npdata_prepath='../DATA/EAC_feat/'
    lstm_data=np.zeros((data_size,24,2048))
    lstm_lb=np.zeros((data_size,12))
    for i,f in enumerate(fls):
        fname,flabel,fpos=f.split('->')
        lstm_lb[i,:]=np.array(patt.findall(flabel))
        
        for t in range(12):
            lstm_lb[i,t]=min(lstm_lb[i,t],1)

        img_name=fls[i].split('.')[0]
        ind_cur=dic[img_name]
        new_fls=gen_ind(img_name)
        for j,f in enumerate(new_fls):
            frame_array=np.load(npdata_prepath+f.split('.')[0]+'.npy')
            lstm_data[i,j,:]=frame_array
    lstm_data=lstm_data.astype('float32')
    lstm_lb=lstm_lb.astype('float32')
    return lstm_data,lstm_lb

# listtrainpath='../DATA/BP4D_10fold/BP4D_SAD_trag_10fd2.txt'
# listtestpath='../DATA/BP4D_10fold/BP4D_SAD_ts_10fd2.txt'

listtrainpath='../DATA/BP4D_SAD_ag_tr.txt'
listtestpath='../DATA/BP4D_SAD_ts.txt'

fp=open(listtrainpath)
imglist=fp.readlines()

#reading test list,ixx contain all the test image names
ft=open(listtestpath)
ixx=ft.readlines()

out_fls=gen_ind(ixx[7500].split('.')[0])
for i in range(len(out_fls)):
    print out_fls[i].split('.')[0]
print len(dic.keys())
trdata,trlb=LSTM_input(ixx)

print trdata[0,0,:],trlb[0,:]

print 'Began training'

for epoch in range(500):
    for batch in range(20):
        loss = train_batch()
        # print loss
    print 'epoch ',epoch, ',train loss is ', loss
    
    loss = test_batch()  
    print epoch,'TEST Loss :', loss
np.savez('data/LSTM1_retrain_fusion_model.npz', *lasagne.layers.get_all_param_values(lstm_net['fc']))

cnt=0
all_predicts=[]
all_labels=[]
ttnum=len(ixx)/BATCH_SIZE
for chunk in batches(ixx, BATCH_SIZE):
    #got all the data based on index
    # print 'finish', cnt ,' of ', ttnum
    tsdata,tslb=LSTM_input(chunk)
    scores = pred_LSfn(tsdata)
    # print scores[0,:]
    # print tslb[0,:]
    num_sc=scores.shape[0]
    for i in range(num_sc):
        all_predicts+=[scores[i,:]]
        all_labels+=[tslb[i,:]]
    cnt+=1
    acc,f1=get_f1_acc(scores,tslb)
    # print acc.mean(),f1.mean()
# print all_predicts.shape
np.savez('data/LSTM1_retrain_result.npz',p=np.array(all_predicts),t=np.array(all_labels))

acc,f1=get_f1_acc_test(np.array(all_predicts),np.array(all_labels))
print acc,f1
print acc.mean(),f1.mean()