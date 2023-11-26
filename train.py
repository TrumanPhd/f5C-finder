from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,roc_auc_score,matthews_corrcoef   
import pandas as pd
import numpy as np
from tensorflow.keras import models,layers,optimizers,regularizers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from utils import *
from Model_lib2 import *
#from Hash1_best import tranformer
from Fusion3feature import tranformer as tranformer3
from Hash1_best import tranformer as hash_net
from encoder1234_best import tranformer as en1234
from evaluation_utils import *

model_dict = {1:Onehot_net,2:Binary_net,3:tranformer3,4:hash_net,5:en1234}

epoch_dict = {1:400,2:400,3:80,4:80,5:80}
#epoch_dict = {1:1,2:1,3:1,4:1,5:1}
dataset_num = 1 
k=10
classification_num = 0.5
np.random.seed(6)
x_train1,y,x_test1,y_test = F5c_onehot26(shuffle_data=False,data = dataset_num)
x_train2,a,x_test2,a = F5c_binary26(shuffle_data=False,data = dataset_num)
x_train4,a,x_test4,a = F5c_encoder(hash1,data = dataset_num)
x_train6,a,x_test6,a = F5c_encoder(condon,data = dataset_num)
x_train5,a,x_test5,a = F5c_1234(shuffle_data=False,data = dataset_num)
x_train3= np.concatenate([x_train4, x_train6, x_train5], axis=1)
x_test3 = np.concatenate([x_test4, x_test6, x_test5], axis=1)
num=len(y)
mode1=np.arange(num/2)%k
mode2=np.arange(num/2)%k
np.random.shuffle(mode1)
np.random.shuffle(mode2)
mode=np.concatenate((mode1,mode2))
test_predict_score=np.zeros(num)

for feature_model in range(3,6):
    def train_and_test():
        model_choose = model_dict[feature_model]
        x = eval('x_train'+str(feature_model))
        #k-cv
        for fold in range(k):
            #split
            trainLabel = y[mode!=fold]
            testLabel  = y[mode==fold]
            trainFeature=x[mode!=fold]
            testFeature=x[mode==fold]
            m1 = model_choose()
            m1.fit(trainFeature,trainLabel,batch_size=256,epochs=epoch_dict[feature_model],verbose=0,validation_data=(testFeature,testLabel),
            shuffle=True)
            test_predict_score[mode==fold]=m1.predict(testFeature).reshape(len(testFeature))
        np.savez('train'+str(feature_model)+'.npz',test_predict=test_predict_score,label=y)
        #test
        x_test = eval('x_test'+str(feature_model))
        m2 = model_choose()
        m2.fit(x,y,batch_size=256,epochs=epoch_dict[feature_model],verbose=0,shuffle=True)
        m2.save_weights(f"saved_model\model"+str(feature_model)+'on_data'+'.h5')
        test_predict=m2.predict(x_test).reshape(len(x_test))
        np.savez('test'+str(feature_model)+'.npz',test_predict=test_predict,label=y_test)
    
    if feature_model != 4:
        train_and_test()