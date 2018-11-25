from __future__ import print_function
from __future__ import division

from collections import OrderedDict
import os
import sys
import warnings

import argparse
import logging
import h5py as h5
import numpy as np
import pandas as pd
import scipy.io

import six
import csv
from six.moves import range

from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score
from keras.preprocessing import sequence
from keras.optimizers import RMSprop,Adam, SGD
from keras.models import Sequential
from keras.layers.core import  Dropout, Activation, Flatten
from keras.regularizers import l1,l2,l1_l2
from keras.constraints import maxnorm
#from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, Dense, LSTM, Bidirectional
#from keras.utils import plot_model
#from keras.utils.layer_utils import print_layer_shapes
# fix random seed for reproducibility
np.random.seed(1369)

if __name__ == '__main__':

 h5filename = "histonemodKmer_resample_ncl.h5"
 h5file = h5.File(h5filename,'r')
 input_features = h5file['input/H3K27me3_kmer']
 output_H3K27me3 = h5file['output/H3K27me3']
 
 input_features = np.array(input_features,dtype='int8')
 output_H3K27me3 = np.array(output_H3K27me3, dtype='int8')
 output_H3K27me3_reshape = output_H3K27me3.reshape(len(output_H3K27me3),1)
 
 #combine the label with input dna
 input_features_label = np.concatenate((input_features,output_H3K27me3_reshape), axis=1)
 H3K27me3_df = pd.DataFrame(output_H3K27me3)
 pos_label= H3K27me3_df.loc[H3K27me3_df.iloc[:,0]==1]
 pos_label_ix = np.array(pos_label.index)
 neg_label = H3K27me3_df.loc[H3K27me3_df.iloc[:,0]==0]
 neg_label_ix = np.array(neg_label.index)
 pos_sam_H3K27me3 = input_features_label[pos_label_ix,:]
 neg_sam_H3K27me3 = input_features_label[neg_label_ix,:]
 print('here')
 print(pos_label_ix)
 print(input_features_label.shape)
 print(pos_label.shape)
 print(neg_label.shape)
 print(pos_sam_H3K27me3.shape)
 print(neg_sam_H3K27me3.shape)
 print(input_features)
 
  # writing data in csv files
 
 pos_arr = pos_sam_H3K27me3[:,0:2080]
 label_str_pos = np.empty(pos_sam_H3K27me3.shape[0],dtype='S4')
 for i in range(0, label_str_pos.shape[0]):
	 label_str_pos[i] = str('TRUE')

 neg_arr = neg_sam_H3K27me3[:,0:2080]
 label_str_neg = np.empty(neg_sam_H3K27me3.shape[0],dtype='S5')
 
 for i in range(0, label_str_neg.shape[0]):
	 label_str_neg[i] = str('FALSE')
	 
 label_str_neg = label_str_neg.reshape(len(label_str_neg),1)
 label_str_pos = label_str_pos.reshape(len(label_str_pos),1)
 pos_arr_csv = np.concatenate((pos_arr,label_str_pos), axis =1)	
 neg_arr_csv = np.concatenate((neg_arr,label_str_neg), axis =1)	
 
 columns = []
 
 for i in range(0,2080):
	 columns.append(str(i))
 
 columns.append('Label')
 
  #columns = ['ATF2', 'BACH1', 'CJUN', 'CMYC', 'E2F6', 'FOSL1', 'NANOG', 'NRSF', 'POU5F1', 'SIN3A',  'SP4',  'TCF12', 'TEAD4',  'ATF3', 'CEBPB', 'CREB', 'CTCF',	'EGR1',	
 #'GABP',	'JUND',	'MAFK',	'MAX',	'RFX5',	'SIX5',	'SP1', 'SRF',	'USF1',	'USF2',	'YY1',	'ZNF274', 'Label' ]
  	 
 print(pos_arr_csv.shape)
 print(neg_arr_csv.shape)
 print(label_str_pos[0:5])
 print(label_str_neg[0:5])
 
 
 print('here')
 #print(pos_label_ix)
 print(input_features_label.shape)
 print(pos_label.shape)
 print(neg_label.shape)
 print(pos_sam_H3K27me3.shape)
 print(neg_sam_H3K27me3.shape)
 
 posfile = 'H3K27me3'+"_"+"kmer_pos_ncl.csv"
 if os.path.exists(posfile):
	 os.remove(posfile)
 fpos = open(posfile,'a')
 negfile = 'H3K27me3'+"_"+"kmer_neg_ncl.csv"
 if os.path.exists(negfile):
	 os.remove(negfile)
 fneg = open(negfile,'a')
 writer = csv.writer(fpos)
 writer.writerow(columns)
 for i in range(0,pos_arr_csv.shape[0]):
	 writer.writerow(pos_arr_csv[i,:])
 fpos.close()
 
 writer2 = csv.writer(fneg)
 writer2.writerow(columns)
 for i in range(0,neg_arr_csv.shape[0]):
	 writer2.writerow(neg_arr_csv[i,:])
 fneg.close()

 
 #train-val- test
 #train
 train_neg_H3K27me3 = neg_sam_H3K27me3[0:16693,:]
 train_pos_H3K27me3 = pos_sam_H3K27me3[0:1729,:]
 train_neg_pos_H3K27me3 = np.concatenate((train_neg_H3K27me3, train_pos_H3K27me3),axis = 0)
 np.random.shuffle(train_neg_pos_H3K27me3)
 X_train_H3K27me3 = train_neg_pos_H3K27me3[:,0:2080]
 Y_train_H3K27me3 = train_neg_pos_H3K27me3[:,2080]
 frq = np.bincount(Y_train_H3K27me3)
 print(frq)
 print(X_train_H3K27me3.shape)
 print(Y_train_H3K27me3.shape)
 
 #val
 val_neg_H3K27me3 = neg_sam_H3K27me3[16693:19077:]
 val_pos_H3K27me3 = pos_sam_H3K27me3 [1729:1976,:]
 val_neg_pos_H3K27me3 = np.concatenate((val_neg_H3K27me3, val_pos_H3K27me3),axis = 0)
 np.random.shuffle(val_neg_pos_H3K27me3)
 X_val_H3K27me3 = val_neg_pos_H3K27me3[:,0:2080]
 Y_val_H3K27me3 = val_neg_pos_H3K27me3[:,2080]
 frq = np.bincount(Y_val_H3K27me3)
 print(frq)
 print(X_val_H3K27me3.shape)
 print(Y_val_H3K27me3.shape)
 
 #test
 test_neg_H3K27me3 = neg_sam_H3K27me3[19077:,:]
 test_pos_H3K27me3 = pos_sam_H3K27me3 [1976:,:]
 test_neg_pos_H3K27me3 = np.concatenate((test_neg_H3K27me3, test_pos_H3K27me3),axis = 0)
 np.random.shuffle(test_neg_pos_H3K27me3)
 X_test_H3K27me3 = test_neg_pos_H3K27me3[:,0:2080]
 Y_test_H3K27me3 = test_neg_pos_H3K27me3[:,2080]
 frq = np.bincount(Y_test_H3K27me3)
 print(frq)
 print(X_test_H3K27me3.shape)
 print(Y_test_H3K27me3.shape)
 
 model = Sequential()
 #model.add(Conv1D(activation="relu", input_shape=(2080, 1), padding="valid", strides=1, filters=256, kernel_size=11, kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.001)))
 #model.add(MaxPooling1D(pool_size=4))
 #model.add(Dropout(0.6))
 #model.add(Conv1D(activation="relu", padding="valid", strides=1, filters=640, kernel_size=3, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.001)))
 #model.add(MaxPooling1D(pool_size=2))
 #model.add(Dropout(0.5))
 #model.add(Flatten())
 #model.summary()
 model.add(Dense(units=512, input_dim=2080, activation="tanh", kernel_initializer='glorot_uniform'))
 model.add(Dropout(0.5))
 #model.add(Dense(units=512, input_dim=512,  activation="relu", kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.001)))
 #model.add(Dropout(0.5))
 model.add(Dense(units=180, activation="tanh",kernel_initializer='glorot_uniform'))
 model.add(Dropout(0.5))
 model.add(Dense(units=70, activation="tanh",kernel_initializer='glorot_uniform'))
 model.add(Dense(units=1, activation="sigmoid"))
 model.summary()
 adam = Adam(lr=0.0001)
 sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
 model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
 print('running at most 60 epochs')
 checkpointer = ModelCheckpoint(filepath="HistoneMark_H3K27me3.hdf5", verbose=1, save_best_only=True)
 earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
 model.fit(X_train_H3K27me3, Y_train_H3K27me3, batch_size=128, epochs=50, shuffle=True, validation_data=( X_val_H3K27me3, Y_val_H3K27me3), callbacks=[checkpointer,earlystopper])
 #model.fit(X_train_s, Y_train_s, batch_size=12, epochs=50, shuffle=True, validation_data=( X_val_s, Y_val_s), callbacks=[checkpointer,earlystopper])
 y_pred = model.predict(X_test_H3K27me3)
 #y_pred = model.predict(X_test_s)
 #tresults = model.evaluate(X_test_s, Y_test_s)
 np.savetxt('H3K27me3_true.csv', Y_test_H3K27me3, delimiter=",")
 np.savetxt('H3K27me3_pred.csv', y_pred, delimiter=",")
 tresults = model.evaluate(X_test_H3K27me3, Y_test_H3K27me3)
 print(tresults)
 model.summary()		
 #print(roc_auc_score(Y_test_s,y_pred))
 print(roc_auc_score(Y_test_H3K27me3, y_pred))
 print(average_precision_score(Y_test_H3K27me3, y_pred))
 y_pred = (y_pred>0.5)
 cm = confusion_matrix(Y_test_H3K27me3, y_pred)
 print(cm)
 
