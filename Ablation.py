from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,roc_auc_score,matthews_corrcoef   
import pandas as pd
import numpy as np
from tensorflow.keras import models,layers,optimizers,regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from utils import *
from Model_lib2 import *
from numpy import mean
#from Hash1_best import tranformer
from Fusion3feature import tranformer as tranformer3
from FusionA import tranformer as tranformerA
from FusionB import tranformer as tranformerB
from FusionC import tranformer as tranformerC
from Hash1_best import tranformer as hash_net
from encoder1234_best import tranformer as en1234
from evaluation_utils import *
from sklearn.model_selection import train_test_split


model_dict = {1:Onehot_net,2:Binary_net,3:tranformer3,5:hash_net,
              6:en1234,7:tranformerA,8:tranformerB,9:tranformerC }

epoch_dict = {1:400,2:400,3:80,4:80,5:80,6:80,7:80,8:80,9:80}
epoch_dict = {1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1}

dataset_num = 1 
k=10
classification_num = 0.5
np.random.seed(6)
x_train1,y,x_test1,y_test = F5c_onehot26(shuffle_data=False,data = dataset_num)
x_train2,a,x_test2,a = F5c_binary26(shuffle_data=False,data = dataset_num)
x_train5,a,x_test5,a = F5c_encoder(hash1,data = dataset_num)
x_train4,a,x_test4,a = F5c_encoder(condon,data = dataset_num)
x_train6,a,x_test6,a = F5c_1234(shuffle_data=False,data = dataset_num)
x_train3= np.concatenate([x_train4, x_train5, x_train6], axis=1)
x_test3 = np.concatenate([x_test4, x_test5, x_test6], axis=1)
x_train7= np.concatenate([x_train6, x_train4], axis=1)
x_test7 = np.concatenate([x_test6, x_test4], axis=1)
x_train8= np.concatenate([x_train6, x_train5], axis=1)
x_test8 = np.concatenate([x_test6, x_test5], axis=1)
x_train9= np.concatenate([x_train5, x_train4], axis=1)
x_test9 = np.concatenate([x_test5, x_test4], axis=1)
num=len(y)
mode1=np.arange(num/2)%k
mode2=np.arange(num/2)%k
np.random.shuffle(mode1)
np.random.shuffle(mode2)
mode=np.concatenate((mode1,mode2))
test_predict_score=np.zeros(num)
fold = 1        
batch_size = 256

for feature_model in range(1,3):
    def train_and_test(dropout_rate):
        model_choose = model_dict[feature_model]
        x = eval('x_train'+str(feature_model))
        X_train, X_test, y_train, y_test = train_test_split(
            x,  # Feature matrix
            y,  # Target variable
            test_size=0.1,  # Percentage of data to allocate to the test set
            random_state=6  # Seed for reproducibility
           )
        m1 = model_choose(dropout=dropout_rate)
        m1.fit(X_train,y_train,batch_size=batch_size,epochs=epoch_dict[feature_model],verbose=0,validation_data=(X_test,y_test),shuffle=True)
        test_predict_score=m1.predict(X_test).reshape(len(X_test))
        np.savez('Ablation/model'+str(feature_model)+'dropout_rate'+str(dropout_rate)+'.npz',test_predict=test_predict_score,label=y_test)

    if feature_model != 4:
        for drop in [0.6,0.5,0.4,0.3,0.2]:
            train_and_test(dropout_rate=drop)


for feature_model in range(4,4):
    def train_and_test(dropout_rate,num_heads):
        model_choose = model_dict[feature_model]
        x = eval('x_train'+str(feature_model))
        X_train, X_test, y_train, y_test = train_test_split(
            x,  # Feature matrix
            y,  # Target variable
            test_size=0.1,  # Percentage of data to allocate to the test set
            random_state=6  # Seed for reproducibility
           )
        m1 = model_choose(dropout=dropout_rate,num_heads=num_heads)
        m1.fit(X_train,y_train,batch_size=batch_size,epochs=epoch_dict[feature_model],verbose=0,validation_data=(X_test,y_test),shuffle=True)
        test_predict_score=m1.predict(X_test).reshape(len(X_test))
        np.savez('Ablation/model'+str(feature_model)+'num_heads'+str(num_heads)+'dropout_rate'+str(dropout_rate)+'.npz',test_predict=test_predict_score,label=y_test)

    if feature_model != 4:
        for drop in [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]:
            for num_heads in [2, 4, 8, 16]:
                
                train_and_test(dropout_rate=drop,num_heads=num_heads)