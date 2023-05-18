#F5c_loader
import numpy as np
from sklearn.utils import shuffle       

def data_chooser(dataset=1):
    folder_path = 'E:/F5C/f5Cfinder/Dataset'
    postrain_path = folder_path+'/postrain' 
    postest_path  = folder_path+'/postest'
    negtrain_path = folder_path+'/negtrain'+str(dataset)
    negtest_path  = folder_path+'/negtest'+str(dataset)
    return postrain_path,postest_path,negtrain_path,negtest_path
    
#For bioinformation
from feature_selection import *
def seq_loader(path):
    seqs = []
    f = open(path)
    for line in f:
        seqs.append(line.strip('\n'))
    f.close()
    return seqs

#data loader with encoder
def F5c_encoder(feature,data=1):
    postrain_path,postest_path,negtrain_path,negtest_path = data_chooser(data)
    x_train = seq_loader(postrain_path)+(seq_loader(negtrain_path))
    x_test  = seq_loader(postest_path)+(seq_loader(negtest_path))
    y_train = np.concatenate((np.ones(1444),np.zeros(1444)))
    y_test = np.concatenate((np.ones(361),np.zeros(361)))

    train = [] 
    test  = []
    for i in range(2888):
        train.append(feature(x_train[i]))
    for i in range(722):
        test.append(feature(x_test[i]))
    train = np.array(train)
    test  = np.array(test)    
    
    return train, y_train, test, y_test
    
def F5c_seq_loader(shuffle_data=True,data=1):
    postrain_path,postest_path,negtrain_path,negtest_path = data_chooser(data)
    x_train = seq_loader(postrain_path)+(seq_loader(negtrain_path))
    x_test  = seq_loader(postest_path)+(seq_loader(negtest_path))
    y_train = np.concatenate((np.ones(1444),np.zeros(1444)))
    y_test = np.concatenate((np.ones(361),np.zeros(361)))
    #from sklearn.utils import shuffle
    if shuffle_data == True: 
        x_train,y_train = shuffle(x_train, y_train, random_state=0)
        x_test,y_test = shuffle(x_test, y_test, random_state=0)
    return x_train, y_train, x_test, y_test
    
def seq_loader1234(path):
    seqs = []
    f = open(path)
    for line in f:
        line1 = line.replace('A','1 ')
        line1 = line1.replace('G','2 ')
        line1 = line1.replace('C','3 ')
        line1 = line1.replace('T','4 ')
        line1 = list(map(int,list(line1.split(' '))[:-1]))
        seqs.append(line1)
    f.close()
    return seqs 
    
def F5c_1234(shuffle_data=True,data=1):
    postrain_path,postest_path,negtrain_path,negtest_path = data_chooser(data)
    x_train = np.array(seq_loader1234(postrain_path)+(seq_loader1234(negtrain_path)))
    x_test  = np.array(seq_loader1234(postest_path)+(seq_loader1234(negtest_path)))
    y_train = np.concatenate((np.ones(1444),np.zeros(1444)))
    y_test = np.concatenate((np.ones(361),np.zeros(361)))
    #from sklearn.utils import shuffle
    if shuffle_data == True: 
        x_train,y_train = shuffle(x_train, y_train, random_state=0)
        x_test,y_test = shuffle(x_test, y_test, random_state=0)
    return x_train, y_train, x_test, y_test

def onehotfeeder(location):
    f = open(location,'r')
    onehot_matrix = []
    for line in f:
        line = line.replace('\n','')
        lengh = len(line)
        onehotline = line.replace('A','1 0 0 0 ')
        onehotline = onehotline.replace('G','0 1 0 0 ')
        onehotline = onehotline.replace('C','0 0 1 0 ')
        onehotline = onehotline.replace('T','0 0 0 1 ')
        onehotline = onehotline.replace('N','0 0 0 0 ')
        onehotline = np.array(onehotline.split(' '))[:-1]
        onehotline = onehotline.reshape(lengh,4)
        onehot_matrix.append(onehotline)
    f.close()
    return list(np.array(onehot_matrix).astype(int))

def F5c_onehot26(shuffle_data=True,data=1):
    postrain_path,postest_path,negtrain_path,negtest_path = data_chooser(data)
    x_train = np.array(onehotfeeder(postrain_path)+(onehotfeeder(negtrain_path)))
    x_test  = np.array(onehotfeeder(postest_path)+(onehotfeeder(negtest_path)))
    y_train = np.concatenate((np.ones(1444),np.zeros(1444)))
    y_test = np.concatenate((np.ones(361),np.zeros(361)))
    #from sklearn.utils import shuffle
    if shuffle_data == True: 
        x_train,y_train = shuffle(x_train, y_train, random_state=0)
        x_test,y_test = shuffle(x_test, y_test, random_state=0)
    return x_train, y_train, x_test, y_test
    
def F5c_binary26(shuffle_data=True,data=1):
    postrain_path,postest_path,negtrain_path,negtest_path = data_chooser(data)
    x_train = seq_loader(postrain_path)+(seq_loader(negtrain_path))
    x_test  = seq_loader(postest_path)+(seq_loader(negtest_path))
    y_train = np.concatenate((np.ones(1444),np.zeros(1444)))
    y_test = np.concatenate((np.ones(361),np.zeros(361)))

    train_Binary = [] 
    test_Binary  = []
    for i in range(2888):
        train_Binary.append(binary_code_2D(x_train[i]))
    for i in range(722):
        test_Binary.append(binary_code_2D(x_test[i]))
    x_train = np.array(train_Binary)
    x_test  = np.array(test_Binary)    

    #from sklearn.utils import shuffle
    if shuffle_data == True: 
        x_train,y_train = shuffle(x_train, y_train, random_state=0)
        x_test,y_test = shuffle(x_test, y_test, random_state=0)
    return x_train, y_train, x_test, y_test



