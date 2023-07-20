from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,roc_auc_score,matthews_corrcoef   
import pandas as pd
import numpy as np
from tensorflow.keras import models,layers,optimizers,regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from Hao_keras_toolkit import *
from Model_lib2 import *

#from Hash1_best import tranformer
from Fusion3feature import tranformer as tranformer3
from FusionA import tranformer as tranformerA
from FusionB import tranformer as tranformerB
from FusionC import tranformer as tranformerC
from Hash1_best import tranformer as hash_net
from encoder1234_best import tranformer as en1234
from numpy import mean
model_dict = {1:Onehot_net_uncompiled(),2:Binary_net_uncompiled(),3:tranformer3_uncompiled82(),5:tranformer3_uncompiled4(max_words=100),
              6:tranformer3_uncompiled4(top_words=5,max_words=101),7:tranformer3_uncompiled4(max_words=106),8:tranformer3_uncompiled4(max_words=201),9:tranformer3_uncompiled81(max_words=105) }
k = 5
class RES():
    def __init__(self):
        self.ACC = []
        self.Sn  = []
        self.Sp  = []
        self.F1  = []
        self.MCC = []
        self.AUC = []
        self.AUC1= []
        self.AUC2= []
        self.AUC3= []
        self.AUC4= []
        self.AUC5= []
        self.AUC6= []
        self.AUC7= []
        self.AUC8= []
        self.AUC9= []
        self.AUC10=[]
    def res_save(self,save_file_path='results.xls'):
        output=open(save_file_path,'w+',encoding='gbk')
        output.write('Acc\tSn\tSp\tF1\tMCC\tAUC\n')
        output.write(str(mean(self.ACC)))
        output.write('\t')
        output.write(str(mean(self.Sn)))
        output.write('\t')
        output.write(str(mean(self.Sp)))
        output.write('\t')
        output.write(str(mean(self.F1)))
        output.write('\t')
        output.write(str(mean(self.MCC)))  
        output.write('\t')      
        output.write(str(mean(self.AUC)))
        output.write('\n')
        output.write(str(np.std(self.ACC) / np.sqrt(len(self.ACC))))
        output.write('\t')
        output.write(str(np.std(self.Sn) / np.sqrt(len(self.Sn))))
        output.write('\t')
        output.write(str(np.std(self.Sp) / np.sqrt(len(self.Sp))))
        output.write('\t')
        output.write(str(np.std(self.F1) / np.sqrt(len(self.F1))))
        output.write('\t')
        output.write(str(np.std(self.MCC) / np.sqrt(len(self.MCC))))
        output.write('\t')
        output.write(str(np.std(self.AUC) / np.sqrt(len(self.AUC))))
        output.write('\n')                
        output.close()
        
  

def citer(y_test, predict_y_test,process='test'):
    TP=0
    TN=0
    FP=0
    FN=0 
    for i in range(0,722):
        if int(y_test[i])==1 and int(predict_y_test[i])==1:
            TP=TP+1
        elif int(y_test[i])==1 and int(predict_y_test[i])==0:
            FN=FN+1
        elif int(y_test[i])==0 and int(predict_y_test[i])==0:
            TN=TN+1
        elif int(y_test[i])==0 and int(predict_y_test[i])==1:
            FP=FP+1
    Sn=float(TP)/(TP+FN)
    Sp=float(TN)/(TN+FP) 
    y_validation=np.array(y_test,dtype=int)  
    predict_y_test=np.array(predict_y_test,dtype=int) 
    F1=f1_score(y_validation,predict_y_test)
    res_test.ACC.append(float(accuracy_score(y_test, predict_y_test)))
    res_test.Sn.append(Sn)
    res_test.Sp.append(Sp)
    res_test.F1.append(F1)
    res_test.MCC.append(float(matthews_corrcoef(y_test,predict_y_test)))

epoch_dict = {1:400,2:400,3:80,4:80,5:80,6:80,7:80,8:80,9:80}
#epoch_dict = {1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1}

res_cv   = RES()
res_test = RES()  


for dataset_num in range(1,11):
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
    tcombine3 = np.zeros(722) 
    testciter = np.load('test99.npz')
    names = locals()
    #test
    for j in range(1,11):
        for ii in range(1,10):
            if ii != 4: 
                m = model_dict[ii]
                m.load_weights('E:\\F5C\\f5Cfinder\\deeplearning\\saved_model\\model'+str(ii)+'on_data'+str(j)+'.h5')
                prob_predict_y_test = m.predict(eval('x_test'+str(ii)))
                prob_predict_y_test = prob_predict_y_test.reshape(722)
                tcombine3 += prob_predict_y_test

    tcombine3 /= 80
    test_predict=[]
    for jj in tcombine3:
        if jj>0.5:
            test_predict.append(1)
        else:
            test_predict.append(0)
                
    citer(testciter['label'],test_predict)
    res_test.AUC.append(roc_auc_score(testciter['label'],tcombine3))
    print("testing data "+str(dataset_num)+" is finished!")

    lw = 2
    tfprcombine,ttprcombine,_=roc_curve(testciter['label'],tcombine3)
    plt.plot()
    plt.title("ROC curve-f5C-finder",fontsize=20)
    plt.plot(tfprcombine, ttprcombine,lw=lw,label='on data'+str(dataset_num)+' AUC = {:.3f}'.format(float(roc_auc_score(testciter['label'],tcombine3))))
    res_test.AUC.append(roc_auc_score(testciter['label'],tcombine3))
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.legend(loc="lower right")
plt.savefig("roc_curve f5C-finder",dpi=1000,bbox_inches = 'tight')
res_test.res_save(save_file_path = 'Fusion_test_result.xls') 

