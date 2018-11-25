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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score
from keras.preprocessing import sequence
from keras.optimizers import RMSprop,Adam, Adadelta, Nadam, Adamax, SGD, Adagrad
from keras.models import Sequential
from keras.layers.core import  Dropout, Activation, Flatten
from keras.regularizers import l1,l2,l1_l2
from keras.constraints import maxnorm
#from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, Dense, LSTM, Bidirectional
#from keras.utils import plot_model
#from keras.utils.layer_utils import print_layer_shapes

def features_selected_gbc(X_tr, y_tr):
	regr= GradientBoostingClassifier(random_state=0)
	regr.fit(X_tr, y_tr)
	coef = regr.feature_importances_
	parameters = {"coef": coef,
				  "model":regr}
	return parameters

if __name__ == '__main__':
 
 h5filename = "histonemodTF_resample_ncl.h5"
 h5file = h5.File(h5filename,'r')
 input_features = h5file['input/H3K9ac_RPKM']
 output_H3K9ac = h5file['output/H3K9ac']
 
 input_features = np.array(input_features)
 output_H3K9ac = np.array(output_H3K9ac)
 output_H3K9ac_reshape = output_H3K9ac.reshape(len(output_H3K9ac),1)
 
 #combine the label with input dna
 input_features_label = np.concatenate((input_features,output_H3K9ac_reshape), axis=1)
 H3K9ac_df = pd.DataFrame(output_H3K9ac)
 pos_label= H3K9ac_df.loc[H3K9ac_df.iloc[:,0]==1]
 pos_label_ix = np.array(pos_label.index)
 neg_label = H3K9ac_df.loc[H3K9ac_df.iloc[:,0]==0]
 neg_label_ix = np.array(neg_label.index)
 pos_sam_H3K9ac = input_features_label[pos_label_ix,:]
 neg_sam_H3K9ac = input_features_label[neg_label_ix,:]
 print('here')
 print(pos_label_ix)
 print(input_features_label.shape)
 print(pos_label.shape)
 print(neg_label.shape)
 print(pos_sam_H3K9ac.shape)
 print(neg_sam_H3K9ac.shape)
 
 '''
 # writing data in csv files
 
 pos_arr = pos_sam_H3K9ac[:,0:30]
 label_str_pos = np.empty(pos_sam_H3K9ac.shape[0],dtype='S4')
 for i in range(0, label_str_pos.shape[0]):
	 label_str_pos[i] = str('TRUE')

 neg_arr = neg_sam_H3K9ac[:,0:30]
 label_str_neg = np.empty(neg_sam_H3K9ac.shape[0],dtype='S5')
 
 for i in range(0, label_str_neg.shape[0]):
	 label_str_neg[i] = str('FALSE')
	 
 label_str_neg = label_str_neg.reshape(len(label_str_neg),1)
 label_str_pos = label_str_pos.reshape(len(label_str_pos),1)
 pos_arr_csv = np.concatenate((pos_arr,label_str_pos), axis =1)	
 neg_arr_csv = np.concatenate((neg_arr,label_str_neg), axis =1)	
 
 columns = ['ATF2', 'BACH1', 'CJUN', 'CMYC', 'E2F6', 'FOSL1', 'NANOG', 'NRSF', 'POU5F1', 'SIN3A',  'SP4',  'TCF12', 'TEAD4',  'ATF3', 'CEBPB', 'CREB', 'CTCF',	'EGR1',	
 'GABP',	'JUND',	'MAFK',	'MAX',	'RFX5',	'SIX5',	'SP1', 'SRF',	'USF1',	'USF2',	'YY1',	'ZNF274', 'Label' ]
 
 
  	 
 print(pos_arr_csv.shape)
 print(neg_arr_csv.shape)
 print(label_str_pos[0:5])
 print(label_str_neg[0:5])
 
 
 print('here')
 #print(pos_label_ix)
 print(input_features_label.shape)
 print(pos_label.shape)
 print(neg_label.shape)
 print(pos_sam_H3K9ac.shape)
 print(neg_sam_H3K9ac.shape)
 
 posfile = 'H3K9ac'+"_"+"TF_pos_ncl.csv"
 if os.path.exists(posfile):
	 os.remove(posfile)
 fpos = open(posfile,'a')
 negfile = 'H3K9ac'+"_"+"TF_neg_ncl.csv"
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
 '''
 
 
 
 #preparing training, validation and testing (70-10-20):
 #trainset
 
 train_neg_H3K9ac = neg_sam_H3K9ac[0:12701,:]
 train_pos_H3K9ac = pos_sam_H3K9ac [0:5040,:]
 train_neg_pos_H3K9ac = np.concatenate((train_neg_H3K9ac, train_pos_H3K9ac),axis = 0)
 np.random.shuffle(train_neg_pos_H3K9ac)
 X_train_H3K9ac = train_neg_pos_H3K9ac[:,0:30]
 Y_train_H3K9ac = train_neg_pos_H3K9ac[:,30]
 Y_train_H3K9ac = np.array(Y_train_H3K9ac, dtype='int8')
 frq = np.bincount(Y_train_H3K9ac)
 print(frq)
 print(X_train_H3K9ac.shape)
 print(Y_train_H3K9ac.shape)
 print(Y_train_H3K9ac)
 
 # feature selection using gbc
 param = features_selected_gbc(X_train_H3K9ac, Y_train_H3K9ac)
 coef = param["coef"]
 print(np.count_nonzero(coef))
 print(coef)
 features = (coef!=0)
 features = coef>=0.03
 #print(indx)
 #trainset
 X_train_H3K9ac = X_train_H3K9ac[:, features]
 
 
 #validation
 
 val_neg_H3K9ac = neg_sam_H3K9ac[12701:14515,:]
 val_pos_H3K9ac = pos_sam_H3K9ac [5040:5760,:]
 val_neg_pos_H3K9ac = np.concatenate((val_neg_H3K9ac, val_pos_H3K9ac),axis = 0)
 np.random.shuffle(val_neg_pos_H3K9ac)
 X_val_H3K9ac = val_neg_pos_H3K9ac[:,0:30]
 Y_val_H3K9ac = val_neg_pos_H3K9ac[:,30]
 Y_val_H3K9ac = np.array(Y_val_H3K9ac, dtype='int8')
 frq = np.bincount(Y_val_H3K9ac)
 X_val_H3K9ac = X_val_H3K9ac[:,features]
 print(frq)
 print(X_val_H3K9ac.shape)
 print(Y_val_H3K9ac.shape)
 
 #test
 
 test_neg_H3K9ac = neg_sam_H3K9ac[14515:,:]
 test_pos_H3K9ac = pos_sam_H3K9ac [5760:,:]
 test_neg_pos_H3K9ac = np.concatenate((test_neg_H3K9ac, test_pos_H3K9ac),axis = 0)
 np.random.shuffle(test_neg_pos_H3K9ac)
 X_test_H3K9ac = test_neg_pos_H3K9ac[:,0:30]
 Y_test_H3K9ac = test_neg_pos_H3K9ac[:,30]
 Y_test_H3K9ac = np.array(Y_test_H3K9ac, dtype='int8')
 frq = np.bincount(Y_test_H3K9ac)
 gbcmodel = param["model"]
 y_pred = gbcmodel.predict(X_test_H3K9ac)
 print('GBC model')
 print(roc_auc_score(Y_test_H3K9ac, y_pred))
 print(average_precision_score(Y_test_H3K9ac, y_pred))
 X_test_H3K9ac = X_test_H3K9ac[:,features]
 print(frq)
 print(X_test_H3K9ac.shape)
 print(Y_test_H3K9ac.shape) 
 print(Y_test_H3K9ac)
 
 
 #building model
 model = Sequential()
 #model.add(Bidirectional(LSTM(30, return_sequences=True),input_shape=(30,1)))
 #model.add(Dropout(0.5))
 #model.add(Flatten())
 #model.summary()
 model.add(Dense(units=256, input_dim=12, activation="relu", kernel_initializer='glorot_uniform')) #180
 model.add(Dropout(0.3))
 model.add(Dense(units=180,  activation="relu",kernel_initializer='glorot_uniform')) #75
 model.add(Dropout(0.3))
 model.add(Dense(units=60,  activation="relu",kernel_initializer='glorot_uniform')) #64
 #model.add(Dropout(0.1))
 model.add(Dense(units=1,  activation="sigmoid"))  
 #adam = SGD(lr=0.00001, momentum=0.99, nesterov=True)
 #adam = Adagrad(lr = 0.01)
 adam = Adam(lr = 0.0001)
 #adam = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
 model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
 print('running at most 60 epochs')
 checkpointer = ModelCheckpoint(filepath="HistoneMark_H3K9ac_TF_ncl.hdf5", verbose=1, save_best_only=True)
 earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
 model.fit(X_train_H3K9ac, Y_train_H3K9ac, batch_size=10, epochs=90, shuffle=True, validation_data=( X_val_H3K9ac, Y_val_H3K9ac), callbacks=[checkpointer,earlystopper])
 #model.fit(X_train_s, Y_train_s, batch_size=12, epochs=50, shuffle=True, validation_data=( X_val_s, Y_val_s), callbacks=[checkpointer,earlystopper])
 y_pred = model.predict(X_test_H3K9ac)
 #y_pred = model.predict(X_test_s)
 #tresults = model.evaluate(X_test_s, Y_test_s)
 np.savetxt('H3K9ac_true.csv', Y_test_H3K9ac, delimiter=",")
 np.savetxt('H3K9ac_pred.csv', y_pred, delimiter=",")
 tresults = model.evaluate(X_test_H3K9ac, Y_test_H3K9ac)
 print(tresults)
 model.summary()		
 #print(roc_auc_score(Y_test_s,y_pred))
 print(roc_auc_score(Y_test_H3K9ac, y_pred))
 print(average_precision_score(Y_test_H3K9ac, y_pred))
 y_pred = (y_pred>0.5)
 cm = confusion_matrix(Y_test_H3K9ac, y_pred)
 print(cm)
 
