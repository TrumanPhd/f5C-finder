# -*- coding: utf-8 -*-
"""
Model_lib2

@author: Truman
"""
from tensorflow.python.keras.layers import *
from tensorflow.keras import models, layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models,layers,optimizers,regularizers
from keras.layers import BatchNormalization
import keras
import tensorflow as tf
#customization adam is better
#selected model with best performance on different features
def Binary_net(max_len=101,depth=3,l1=64,l2=512,l3=256,gamma=1e-4,lr=1e-4,dropout=0.5,activation='tanh'):
    model=models.Sequential()
    model.add(layers.LSTM(l1,activation=activation,return_sequences=True,kernel_regularizer=regularizers.l1(gamma),input_shape=(max_len,depth)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(l2,activation='tanh',kernel_regularizer=regularizers.l2(gamma)))    
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(gamma)))
    adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=['acc'])
    return model

def Onehot_net(max_len=101,depth=4,l1=64,l2=512,l3=256,gamma=1e-4,lr=1e-4,dropout=0.5,activation='tanh'):
    model=models.Sequential()
    model.add(layers.LSTM(l1,activation=activation,return_sequences=True,kernel_regularizer=regularizers.l1(gamma),input_shape=(max_len,depth)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(l2,activation='tanh',kernel_regularizer=regularizers.l2(gamma)))    
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(gamma)))
    adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=['acc'])
    return model

#uncompiled model 
def Binary_net_uncompiled(max_len=101,depth=3,l1=64,l2=512,l3=256,gamma=1e-4,lr=1e-4):
    model=models.Sequential()
    model.add(layers.LSTM(l1,activation='tanh',return_sequences=True,kernel_regularizer=regularizers.l1(gamma),input_shape=(max_len,depth)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(l2,activation='tanh',kernel_regularizer=regularizers.l2(gamma)))    
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(gamma)))
    return model

def Onehot_net_uncompiled(max_len=101,depth=4,l1=64,l2=512,l3=256,gamma=1e-4,lr=1e-4):
    model=models.Sequential()
    model.add(layers.LSTM(l1,activation='tanh',return_sequences=True,kernel_regularizer=regularizers.l1(gamma),input_shape=(max_len,depth)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(l2,activation='tanh',kernel_regularizer=regularizers.l2(gamma)))    
    model.add(layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l2(gamma)))
    return model

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
        self.attention_weights = None
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    def get_attention_weights(self):
        return self.attention_weights
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
import numpy as np
show_confusion_matrix=True
show_loss=True
np.random.seed(0)  
top_words=50   #feature number
max_words=106  #seq-len
embed_dim=32    
num_labels=1      

def tranformer_demo(top_words=50,max_words=106,num_labels=num_labels,hidden_dim=[64]):
    'PositionalEmbedding+Transformer'
    inputs = Input(name='inputs',shape=[max_words,], dtype='float64')
    x= PositionalEmbedding(sequence_length=max_words, input_dim=top_words, output_dim=embed_dim)(inputs)
    x = TransformerEncoder(32, 32, 4)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_labels, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model

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
top_words=100  #feature number
max_words=206  #seq-len
embed_dim=64    
num_labels=1      
    

def tranformer3(top_words=100,max_words=206,num_labels=num_labels,hidden_dim=[64]):
    'PositionalEmbedding+Transformer'
    inputs = Input(name='inputs',shape=[max_words,], dtype='float64')
    x= PositionalEmbedding(sequence_length=max_words, input_dim=top_words, output_dim=embed_dim)(inputs)
    x = TransformerEncoder(64, 32, 8)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_labels, activation='sigmoid')(x)
    model = Model(inputs, outputs)
        
    adam = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=['acc'])
    return model

def tranformer3_uncompiled82(top_words=100,max_words=206,num_labels=num_labels,hidden_dim=[64]):
    'PositionalEmbedding+Transformer'
    inputs = Input(name='inputs',shape=[max_words,], dtype='float64')
    x= PositionalEmbedding(sequence_length=max_words, input_dim=top_words, output_dim=embed_dim)(inputs)
    x = TransformerEncoder(64, 32, 8)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_labels, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model

def tranformer3_uncompiled81(top_words=100,max_words=206,num_labels=num_labels,hidden_dim=[64]):
    'PositionalEmbedding+Transformer'
    inputs = Input(name='inputs',shape=[max_words,], dtype='float64')
    x= PositionalEmbedding(sequence_length=max_words, input_dim=top_words, output_dim=32)(inputs)
    x = TransformerEncoder(32, 32, 8)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_labels, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model

def tranformer3_uncompiled4(top_words=100,max_words=206,num_labels=num_labels,hidden_dim=[64]):
    'PositionalEmbedding+Transformer'
    inputs = Input(name='inputs',shape=[max_words,], dtype='float64')
    x= PositionalEmbedding(sequence_length=max_words, input_dim=top_words, output_dim=32)(inputs)
    x = TransformerEncoder(32, 32, 4)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_labels, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model

