import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models,layers,optimizers,regularizers
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.utils import shuffle 
import matplotlib.pyplot as plt
from numpy import mean as mean
from feature_selection import *
from Hao_keras_toolkit import *
from tensorflow.keras.callbacks import Callback
#PLOT
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


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),layers.Dense(embed_dim),] )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
 
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
 
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim, })
        return config
    
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
 
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
 
    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
 
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,})
        return config

from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Dense,Input, Dropout, Embedding, Flatten,MaxPooling1D,Conv1D,SimpleRNN,LSTM,GRU,Multiply,GlobalMaxPooling1D
from keras.layers import Bidirectional,Activation,BatchNormalization,GlobalAveragePooling1D,MultiHeadAttention
from keras.callbacks import EarlyStopping
from keras.layers.merge import concatenate
import numpy as np
show_confusion_matrix=True
show_loss=True
np.random.seed(0)  
top_words=100   
max_words=100  
embed_dim=32    
num_labels=1      
   
def tranformer(top_words=top_words,max_words=max_words,num_labels=num_labels,hidden_dim=[64]):
    'PositionalEmbedding+Transformer'
    inputs = Input(name='inputs',shape=[max_words,], dtype='float64')
    x= PositionalEmbedding(sequence_length=max_words, input_dim=top_words, output_dim=embed_dim)(inputs)
    x = TransformerEncoder(embed_dim, 32, 4)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_labels, activation='sigmoid')(x)
    model = Model(inputs, outputs)
        
    adam = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=['acc'])
    return model 

def fivecv(max_words=max_words,batch_size=256,epochs=50,hidden_dim=[32],show_loss=True,show_confusion_matrix=True):
    X, y, unuse1, unuse2 = F5c_encoder(hash1)
    fold_path='E:\\F5C\\f5Cfinder\\Dataset\\fold_five_1.txt'
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
        model=tranformer(max_words=max_words)                     
        auc = RocAucEvaluation(validation_data=(test_sequence,test_label), interval=1)
        hsit = model.fit(train_sequence, train_label, epochs = epochs, batch_size = 256, validation_data=(test_sequence, test_label),callbacks=[auc],verbose=0,shuffle=True)
        score = model.evaluate(test_sequence, test_label, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        if   i == 1:
            auc_plt = auc.train_auc_plot_start()
        elif i == 5: 
            auc_plt = auc.train_auc_plot_end(auc_plt = auc_plt,path_name="hash1")
        else:
            auc.train_auc_plot(auc_plt = auc_plt)
        print(str(i)+"/5finished!")
    print("5cv finished!")

#fivecv(epochs=300)

