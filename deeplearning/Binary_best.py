#f5c deeplearning
#DNA one_hot_encoding
"""
@author: Truman
"""
import sys
import numpy as np
from tensorflow.python.keras.callbacks import Callback
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.utils import shuffle 
import matplotlib.pyplot as plt
from numpy import mean as mean
from Model_lib2 import *
from feature_selection import *
from Hao_keras_toolkit import *
################################################################
model_name = Binary_net

#5cv 
def FiveCV(X,y,path_name=model_name):
    fold_path=r'E:\F5C\f5Cfinder\Dataset\fold_five_1.txt'
    fold_auc=[]
    fold=[]
    fold_file=open(fold_path,'r')
    for line in fold_file:
        fold_temp=line.split()
        fold.append(fold_temp[0])
    fold_file.close()
    num_folds=5
    for i in range(1,num_folds+1):
        X_train=[]
        X_test=[]
        y_train=[]
        y_test=[]
        combined_prob=[]
        for j in range(0,len(fold)):
            if int(fold[j])==i:
                X_test.append(X[j])
                y_test.append(y[j])
            else:
                X_train.append(X[j])
                y_train.append(y[j])
        
        train_sequence = np.array(X_train)
        train_label    = np.array(y_train)
        test_sequence  = np.array(X_test)
        test_label     = np.array(y_test)       
        #network and training
        model = path_name()                     
        auc = RocAucEvaluation(validation_data=(test_sequence,test_label), interval=1)
        hsit = model.fit(train_sequence, train_label, epochs = 500, batch_size = 256, validation_data=(test_sequence, test_label),callbacks=[auc],verbose=0,shuffle=True)
        if   i == 1:
            auc_plt = auc.train_auc_plot_start()
        elif i == 5: 
            auc_plt = auc.train_auc_plot_end(auc_plt = auc_plt,path_name=(str(path_name).split(" "))[1])
        else:
            auc.train_auc_plot(auc_plt = auc_plt)
        print(str(i)+"/5finished!")
    print("5cv finished!")
    


#plot AUC curve
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.x_val,self.y_val = validation_data
        self.roc_auc = []
        self.epoch_num = 0
    def on_epoch_end(self, epoch, log={}):
        if epoch % self.interval == 0:
            self.epoch_num += 1
            y_pred = self.model.predict(self.x_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            self.roc_auc.append(score)
            print('\n ROC_AUC - epoch:%d - auc:%.6f \n' % (epoch+1, score))
        #x_train,y_train,x_label,y_label = train_test_split(train_feature, train_label, train_size=0.95, random_state=233)
    def train_auc_plot_start(self):
        iters = range(self.epoch_num)
        #plt.grid(alpha=0.2) 
        # acc
        plt.plot(iters, self.roc_auc, 'r', label='5cv auc')
        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('5cv auc')
        plt.legend(loc="lower right")
        #plt.show()
        #res.savefig("E:\F5C\\f5Cfinder\\deeplearning\\auc_plot\\FiveCV.png") 
        return plt        
    def train_auc_plot(self,auc_plt):
        iters = range(self.epoch_num)
        auc_plt.plot(iters, self.roc_auc, 'r', label='5cv auc')
        return plt
    def train_auc_plot_end(self,auc_plt,path_name):
        iters = range(self.epoch_num)
        auc_plt.plot(iters, self.roc_auc, 'r', label='5cv auc')
        plt.savefig("E:\F5C\\f5Cfinder\\deeplearning\\combine\\5cross_validation"+str(path_name)+".png") 

"-----------------------------main----------------------"

def main(argv = sys.argv):
    x_train, y_train, x_test, y_test = F5c_binary26()   

    #from sklearn.utils import shuffle
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_test, y_test = shuffle(x_test, y_test, random_state=0)     
   
    #5cv
    FiveCV(x_train,y_train,model_name)
       
if __name__=='__main__':
    main()
