from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,roc_auc_score,matthews_corrcoef   
import pandas as pd
import numpy as np
from tensorflow.keras import models,layers,optimizers,regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from Hao_keras_toolkit import *
from Model_lib2 import *
from numpy import mean
#from Hash1_best import tranformer
from Fusion3feature import tranformer as tranformer3
from FusionA import tranformer as tranformerA
from FusionB import tranformer as tranformerB
from FusionC import tranformer as tranformerC
from Hash1_best import tranformer as hash_net
from encoder1234_best import tranformer as en1234

classification_num = 0.5

model_dict = {1:Onehot_net,2:Binary_net,3:tranformer3,5:hash_net,
              6:en1234,7:tranformerA,8:tranformerB,9:tranformerC }

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
        
res_cv   = RES()
res_test = RES()     

def citer(y_test, predict_y_test,process='test'):
    TP=0
    TN=0
    FP=0
    FN=0 
    for i in range(0,len(y_test)):
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
    
    if process != 'test':
        #mean of k fold validation   
        res_cv.ACC.append(float(accuracy_score(y_test, predict_y_test)))
        res_cv.Sn.append(Sn)
        res_cv.Sp.append(Sp)
        res_cv.F1.append(F1)
        res_cv.MCC.append(float(matthews_corrcoef(y_test,predict_y_test)))
    else:
        #print("indepandent test Sn:%.3lf"%Sn)
        #print("indepandent test Sp:%.3lf"%Sp)
        #print("indepandent test F1:%.3lf"%F1)
        res_test.ACC.append(float(accuracy_score(y_test, predict_y_test)))
        res_test.Sn.append(Sn)
        res_test.Sp.append(Sp)
        res_test.F1.append(F1)
        res_test.MCC.append(float(matthews_corrcoef(y_test,predict_y_test)))

epoch_dict = {1:400,2:400,3:80,4:80,5:80,6:80,7:80,8:80,9:80}
#epoch_dict = {1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1}
for dataset_num in range(1,11):
    k=5
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
    
    for feature_model in range(1,4):

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
            np.savez('train'+str(feature_model)+str(dataset_num)+'.npz',test_predict=test_predict_score,label=y)
            #test
            x_test = eval('x_test'+str(feature_model))
            m2 = model_choose()
            m2.fit(x,y,batch_size=256,epochs=epoch_dict[feature_model],verbose=0,shuffle=True)
            #m2.save("E:\\F5C\\f5Cfinder\\deeplearning\\saved_model\\"+str(feature_model))
            m2.save_weights("E:\\F5C\\f5Cfinder\\deeplearning\\saved_model\\model"+str(feature_model)+'on_data'+str(dataset_num)+'.h5')
            test_predict=m2.predict(x_test).reshape(len(x_test))
            np.savez('test'+str(feature_model)+str(dataset_num)+'.npz',test_predict=test_predict,label=y_test)
        #train_and_test()
        
    for feature_model in range(5,10):

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
            np.savez('train'+str(feature_model)+str(dataset_num)+'.npz',test_predict=test_predict_score,label=y)
            #test
            x_test = eval('x_test'+str(feature_model))
            m2 = model_choose()
            m2.fit(x,y,batch_size=256,epochs=epoch_dict[feature_model],verbose=0,shuffle=True)
            #m2.save("E:\\F5C\\f5Cfinder\\deeplearning\\saved_model\\"+str(feature_model))
            m2.save_weights("E:\\F5C\\f5Cfinder\\deeplearning\\saved_model\\model"+str(feature_model)+'on_data'+str(dataset_num)+'.h5')
            test_predict=m2.predict(x_test).reshape(len(x_test))
            np.savez('test'+str(feature_model)+str(dataset_num)+'.npz',test_predict=test_predict,label=y_test)
        #train_and_test()
        
    names = locals()
    combine3 = np.zeros(2888) 
    train1 = np.load('train99.npz')
    for i in range(1,len(model_dict)+1):
        if i != 4:
            combine3 += np.load('train'+str(i)+str(dataset_num)+'.npz')['test_predict']
    combine3 = combine3/8
    fprcombine,tprcombine,_=roc_curve(train1['label'],combine3)

                          
    train_predict=[]
    for i in combine3:
        if i>classification_num:
            train_predict.append(1)
        else:
            train_predict.append(0)
    
    #print(str(k)+" fold validation acc:%.3lf"%accuracy_score(train1['label'],train_predict))
    #print(str(k)+" fold validation precision:%.3lf"%precision_score(train1['label'],train_predict))
    #print(str(k)+" fold validation recall:%.3lf"%recall_score(train1['label'],train_predict))
    #print(str(k)+" fold validation MCC:%.3lf"%matthews_corrcoef(train1['label'],train_predict))
    citer(train1['label'],train_predict,"train")
    
    
    #test
    tcombine3 = np.zeros(722) 
    test1 = np.load('test99.npz')
    for i in range(1,len(model_dict)+1):
        if i != 4:
            tcombine3 += np.load('test'+str(i)+str(dataset_num)+'.npz')['test_predict']
    tcombine3 /= 8
    tfprcombine,ttprcombine,_=roc_curve(test1['label'],tcombine3)
   
    test_predict=[]
    for i in tcombine3:
        if i>classification_num:
            test_predict.append(1)
        else:
            test_predict.append(0)
            
    #print("indepandent test acc:%.3lf"%accuracy_score(test1['label'],test_predict))
    #print("indepandent test precision:%.3lf"%precision_score(test1['label'],test_predict))
    #print("indepandent test recall:%.3lf"%recall_score(test1['label'],test_predict))
    #print("indepandent test MCC:%.3lf"%matthews_corrcoef(test1['label'],test_predict))
    citer(test1['label'],test_predict)
    
    lw = 1.2
    """    
    #plt.subplot(121)
    plt.plot()
    plt.rcParams.update({"font.size":10})
    plt.title("ROC curve-Combine-5CV",fontsize=20)
    plt.plot(fprcombine, tprcombine,lw=lw,label='on data'+str(dataset_num)+' AUC = {:.3f}'.format(float(roc_auc_score(train1['label'],combine3))))
    res_cv.AUC.append(roc_auc_score(train1['label'],combine3))
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.legend(loc="lower right")
plt.savefig("roc_curve 5CV",dpi=1000,bbox_inches = 'tight')
"""
    #plt.subplot(122)
    plt.plot()
    plt.title("ROC curve-Combine-independent test",fontsize=20)
    plt.plot(tfprcombine, ttprcombine,lw=lw,label='on data'+str(dataset_num)+' AUC = {:.3f}'.format(float(roc_auc_score(test1['label'],tcombine3))))
    res_test.AUC.append(roc_auc_score(test1['label'],tcombine3))
    plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.legend(loc="lower right")
plt.savefig("roc_curve independent test",dpi=1000,bbox_inches = 'tight')
    #"""
res_cv.res_save(save_file_path = 'cv_result.xls')
res_test.res_save(save_file_path = 'test_result.xls')   


