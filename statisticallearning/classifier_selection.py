#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 11:11:20 2018

@author: liuze
"""
import os
import matplotlib.pyplot as plt
from sys import exit
import sys
import joblib
from feature_selection import *
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from numpy import  *
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from deepforest import CascadeForestClassifier
def exit_with_help(argv):
	print("""\
Usage: python {0} positive_dataset negative_dataset matrix_motif1 matrix_motif2
This script trains five different 6mA prediction models.
Including:
     1--RandomForest
     2--GradientBoosting
     3--AdaBoost
     4--ExtraTrees
     5--SVM
     6--a ensemble classifier with the above five methods. 
5-fold cross-validation was used to evaluate the performance of each classifier.""".format(argv[0]))
	sys.exit()

def process_options(argv):
    argc=len(argv)
    if argc!=3:
        exit_with_help(argv)
    posCV=open(argv[1],'r')
    negCV=open(argv[2],'r')
    #matrix_motif1_file=open(argv[3],'r')
    #matrix_motif2_file=open(argv[4],'r')
    return posCV, negCV
        
def main(argv=sys.argv):
    pos_train_file, neg_train_file=process_options(argv)

######import motif1 matrix    
#    matrix_motif1=[]
#    for line in matrix_motif1_file:
#        matrix_motif1_vector=line.split()
#        matrix_motif1.append(matrix_motif1_vector)
#    matrix_motif1_file.close()
#    matrix1_array_T=list(map(list,zip(*matrix_motif1)))
#    matrix1_array=np.array(matrix1_array_T,dtype=np.float32) 
#
#######import motif2 matrix
#    matrix_motif2=[]
#    for line in matrix_motif2_file:
#        matrix_motif2_vector=line.split()
#        matrix_motif2.append(matrix_motif2_vector)
#    matrix_motif2_file.close()
#    matrix2_array_T=list(map(list,zip(*matrix_motif2)))
#    matrix2_array=np.array(matrix2_array_T,dtype=np.float32)     
######train feature extraction
    feature_matrix=[]
    label_vector=[]
    for line in pos_train_file:
        feature_vector=[]
        sequence_infor=line.split()     
        sequence=sequence_infor[0]
        #feature_vector.extend(ksnpf(sequence)+nucleic_shift(sequence)+binary_code(sequence))
#        matrix1_score=matrix_motif1_func(sequence,matrix1_array)
#        feature_vector.append(matrix1_score)
#        matrix2_score=matrix_motif2_func(sequence,matrix2_array)
#        feature_vector.extend(matrix2_score)
        feature_vector.extend(kmer(sequence)+ksnpf(sequence)+nucleic_shift(sequence)+binary_code(sequence)+EIIP(sequence)+TF_content_location(sequence)+globalDescription(sequence)+FE(sequence)+MMI(sequence)+hash1(sequence)+condon(sequence))
        #feature_vector.extend(binary_code(sequence)+hash1(sequence)+condon(sequence))
        label_vector.append('1')
        feature_matrix.append(feature_vector)
    pos_train_file.close()
    
    for line in neg_train_file:
        feature_vector=[]
        sequence_infor=line.split()
        sequence=sequence_infor[0]
        #feature_vector.extend(ksnpf(sequence)+nucleic_shift(sequence)+binary_code(sequence))
#        matrix1_score=matrix_motif1_func(sequence,matrix1_array)
#        feature_vector.append(matrix1_score)
#        matrix2_score=matrix_motif2_func(sequence,matrix2_array)
#        feature_vector.extend(matrix2_score)
        feature_vector.extend(kmer(sequence)+ksnpf(sequence)+nucleic_shift(sequence)+binary_code(sequence)+EIIP(sequence)+TF_content_location(sequence)+globalDescription(sequence)+FE(sequence)+MMI(sequence)+hash1(sequence)+condon(sequence))
        #feature_vector.extend(binary_code(sequence)+hash1(sequence)+condon(sequence))
        label_vector.append('-1')
        feature_matrix.append(feature_vector)
    feature_array = np.array(feature_matrix,dtype=np.float32)
    min_max_scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-1, 1))
    feature_scaled= min_max_scaler.fit_transform(feature_array)
    neg_train_file.close()

    np.savetxt("all_features.txt", np.array(feature_array,dtype=np.float32))
    np.savetxt("label.txt", np.array(label_vector,dtype=np.int))

    X=feature_scaled
    y=label_vector
####using the optimized feature subset    
    indices_file=open('./indices.txt','r')
    indices=[]
    for line in indices_file:
        line_temp=line.strip().split()
        indices.append(int(line_temp[0]))
    X=X[:,indices[0:196]]
    
    # clf = ExtraTreesClassifier(n_estimators=100)
    # clf = clf.fit(X, y)
    # importances = clf.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in clf.estimators_],axis=0)
    # indices = np.argsort(importances)[::-1]
    # # Print the feature ranking
    # print("Feature ranking:")
    
    # for f in range(X.shape[1]):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # # Plot the feature importances of the forest
    # feature_importance=importances[indices]
    # ranked_feature=list(indices)
    # ks=X[:,ranked_feature]
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(X.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
    # plt.xticks(range(X.shape[1]), indices)
    # plt.xlim([-1, X.shape[1]])
    # plt.show()
    # print(clf.feature_importances_)
#    model = SelectFromModel(clf, prefit=True,threshold=0.002)
#    X= model.transform(X)
#    print(X.shape)
#    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
#    model = SelectFromModel(lsvc, prefit=True)
#    X= model.transform(X) 
    
######GridSearchCV
    
#    C_range = np.logspace(-1, 1, 100)
#    gamma_range = np.logspace(-2, 1, 100)
#    param_grid = dict(gamma=gamma_range, C=C_range)
#    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
#    grid.fit(X, y)
#    print("The best parameters are %s with a score of %0.2f"% (grid.best_params_, grid.best_score_))
    
    #fold_path='./cross_data/fold_five1.txt'
    fold_path='../Dataset/fold_five_8.txt'
    fold_auc=[]
    fold=[]
    fold_file=open(fold_path,'r')
    for line in fold_file:
        fold_temp=line.split()
        fold.append(fold_temp[0])
    fold_file.close()
    num_folds=5
    RandomForest_ACC=[]
    RandomForest_AUC=[]
    RandomForest_Sn=[]
    RandomForest_Sp=[]
    RandomForest_MCC=[]
    RandomForest_F1=[]
    GradientBoosting_ACC=[]
    GradientBoosting_AUC=[]
    GradientBoosting_Sn=[]
    GradientBoosting_Sp=[]
    GradientBoosting_MCC=[]
    GradientBoosting_F1=[]
    AdaBoost_ACC=[]
    AdaBoost_AUC=[]
    AdaBoost_Sn=[]
    AdaBoost_Sp=[]
    AdaBoost_MCC=[]
    AdaBoost_F1=[]    
    ExtraTrees_ACC=[]
    ExtraTrees_AUC=[]
    ExtraTrees_Sn=[]
    ExtraTrees_Sp=[]
    ExtraTrees_MCC=[]
    ExtraTrees_F1=[]    
    SVC_ACC=[]
    SVC_AUC=[]
    SVC_Sn=[]
    SVC_Sp=[]
    SVC_MCC=[]
    SVC_F1=[]     
    combined_ACC=[]
    combined_AUC=[]
    combined_Sn=[]
    combined_Sp=[]
    
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
######randomforestClassifier
        # X_train=np.array(X_train)
        # y_train=np.array(y_train)
        # X_test=np.array(X_test)
        # y_test=np.array(y_test)        
        # clf = CascadeForestClassifier(random_state=1)       
        #clf = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_split=2, random_state=0)
        clf = GradientBoostingClassifier(n_estimators=85)
        #clf = AdaBoostClassifier(n_estimators=100)
        #clf = ExtraTreesClassifier(n_estimators=100)
        #clf = SVC(C=0.655,gamma=0.013,probability=True)
        clf.fit(X_train,y_train)
        r = clf.score(X_test,y_test)
        #print('RandomForest Mean Accuracy:%s'%r)
        RandomForest_ACC.append(r)
        predict_y_test = clf.predict(X_test)
        TP=0
        TN=0
        FP=0
        FN=0 
        for i in range(0,len(y_test)):
            if int(y_test[i])==1 and int(predict_y_test[i])==1:
                TP=TP+1
            elif int(y_test[i])==1 and int(predict_y_test[i])==-1:
                FN=FN+1
            elif int(y_test[i])==-1 and int(predict_y_test[i])==-1:
                TN=TN+1
            elif int(y_test[i])==-1 and int(predict_y_test[i])==1:
                FP=FP+1
        Sn=float(TP)/(TP+FN)
        Sp=float(TN)/(TN+FP)
        ACC=float((TP+TN))/(TP+TN+FP+FN)
        RandomForest_Sn.append(Sn)
        RandomForest_Sp.append(Sp)        
        
        prob_predict_y_test = clf.predict_proba(X_test)
        predictions_test = prob_predict_y_test[:, 1]
        y_validation=np.array(y_test,dtype=int)
        fpr, tpr, thresholds =metrics.roc_curve(y_validation, predictions_test,pos_label=1)
        roc_auc = auc(fpr, tpr)
        #print('RandomForest AUC:%s'%roc_auc)
        RandomForest_AUC.append(roc_auc)
        F1=metrics.f1_score(y_validation, list(map(int,predict_y_test)))
        MCC=metrics.matthews_corrcoef(y_validation,list(map(int,predict_y_test)))
        RandomForest_F1.append(F1)
        RandomForest_MCC.append(MCC)
        combined_prob.append(predictions_test)
######GradientBoostingClassifier
        # clf = clf = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,max_depth=1, random_state=0)
        # clf.fit(X_train,y_train)
        # r = clf.score(X_test,y_test)
        # #print('GradientBoostingClassifier Mean Accuracy:%s'%r)
        # GradientBoosting_ACC.append(r)
        # predict_y_test = clf.predict(X_test)

        # TP=0
        # TN=0
        # FP=0
        # FN=0 
        # for i in range(0,len(y_test)):
        #     if int(y_test[i])==1 and int(predict_y_test[i])==1:
        #         TP=TP+1
        #     elif int(y_test[i])==1 and int(predict_y_test[i])==-1:
        #         FN=FN+1
        #     elif int(y_test[i])==-1 and int(predict_y_test[i])==-1:
        #         TN=TN+1
        #     elif int(y_test[i])==-1 and int(predict_y_test[i])==1:
        #         FP=FP+1
        # Sn=float(TP)/(TP+FN)
        # Sp=float(TN)/(TN+FP)
        # ACC=float((TP+TN))/(TP+TN+FP+FN)
        # GradientBoosting_Sn.append(Sn)
        # GradientBoosting_Sp.append(Sp)        
        
        # prob_predict_y_test = clf.predict_proba(X_test)
        # predictions_test = prob_predict_y_test[:, 1]
        # y_validation=np.array(y_test,dtype=int)
        # fpr, tpr, thresholds =metrics.roc_curve(y_validation, predictions_test,pos_label=1)
        # roc_auc = auc(fpr, tpr)
        # #print('GradientBoostingClassifier AUC:%s'%roc_auc)
        # GradientBoosting_AUC.append(roc_auc)
        # F1=metrics.f1_score(y_validation, list(map(int,predict_y_test)))
        # MCC=metrics.matthews_corrcoef(y_validation,list(map(int,predict_y_test)))
        # GradientBoosting_F1.append(F1)
        # GradientBoosting_MCC.append(MCC)
        # combined_prob.append(predictions_test)
######AdaBoostClassifier
        # clf = AdaBoostClassifier(n_estimators=10)
        # clf.fit(X_train,y_train)
        # r = clf.score(X_test,y_test)
        # #print('AdaBoostClassifier Mean Accuracy:%s'%r)
        # AdaBoost_ACC.append(r)
        # predict_y_test = clf.predict(X_test)
        
        # TP=0
        # TN=0
        # FP=0
        # FN=0 
        # for i in range(0,len(y_test)):
        #     if int(y_test[i])==1 and int(predict_y_test[i])==1:
        #         TP=TP+1
        #     elif int(y_test[i])==1 and int(predict_y_test[i])==-1:
        #         FN=FN+1
        #     elif int(y_test[i])==-1 and int(predict_y_test[i])==-1:
        #         TN=TN+1
        #     elif int(y_test[i])==-1 and int(predict_y_test[i])==1:
        #         FP=FP+1
        # Sn=float(TP)/(TP+FN)
        # Sp=float(TN)/(TN+FP)
        # ACC=float((TP+TN))/(TP+TN+FP+FN)
        # AdaBoost_Sn.append(Sn)
        # AdaBoost_Sp.append(Sp)     
        
        # prob_predict_y_test = clf.predict_proba(X_test)
        # predictions_test = prob_predict_y_test[:, 1]
#######generate combined negative scores        
        #combined_prob=predictions_test        
        
        # y_validation=np.array(y_test,dtype=int)
        # fpr, tpr, thresholds =metrics.roc_curve(y_validation, predictions_test,pos_label=1)
        # roc_auc = auc(fpr, tpr)
        # #print('AdaBoostClassifier AUC:%s'%roc_auc)
        # AdaBoost_AUC.append(roc_auc)
        # F1=metrics.f1_score(y_validation, list(map(int,predict_y_test)))
        # MCC=metrics.matthews_corrcoef(y_validation,list(map(int,predict_y_test)))
        # AdaBoost_F1.append(F1)
        # AdaBoost_MCC.append(MCC)
        # combined_prob.append(predictions_test)
######extraTreesClassifier
#         clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=1)
#         clf.fit(X_train,y_train)
#         r = clf.score(X_test,y_test)
#         #print('ExtraTreesClassifier Mean Accuracy:%s'%r)
#         ExtraTrees_ACC.append(r)
#         predict_y_test = clf.predict(X_test)
        
#         TP=0
#         TN=0
#         FP=0
#         FN=0 
#         for i in range(0,len(y_test)):
#             if int(y_test[i])==1 and int(predict_y_test[i])==1:
#                 TP=TP+1
#             elif int(y_test[i])==1 and int(predict_y_test[i])==-1:
#                 FN=FN+1
#             elif int(y_test[i])==-1 and int(predict_y_test[i])==-1:
#                 TN=TN+1
#             elif int(y_test[i])==-1 and int(predict_y_test[i])==1:
#                 FP=FP+1
#         Sn=float(TP)/(TP+FN)
#         Sp=float(TN)/(TN+FP)
#         ACC=float((TP+TN))/(TP+TN+FP+FN)
#         ExtraTrees_Sn.append(Sn)
#         ExtraTrees_Sp.append(Sp)                  
        
#         prob_predict_y_test = clf.predict_proba(X_test)
#         predictions_test = prob_predict_y_test[:, 1]
# ######generate combined positive scores        
# #        for j in range(0,len(y_test)):
# #            if int(y_test[j])==1:
# #                combined_prob[j]=predictions_test[j]
                
        
#         y_validation=np.array(y_test,dtype=int)
#         fpr, tpr, thresholds =metrics.roc_curve(y_validation, predictions_test,pos_label=1)
#         roc_auc = auc(fpr, tpr)
#         #print('ExtraTreesClassifier AUC:%s'%roc_auc)
#         ExtraTrees_AUC.append(roc_auc)
#         F1=metrics.f1_score(y_validation, list(map(int,predict_y_test)))
#         MCC=metrics.matthews_corrcoef(y_validation,list(map(int,predict_y_test)))
#         ExtraTrees_F1.append(F1)
#         ExtraTrees_MCC.append(MCC)
#         combined_prob.append(predictions_test)
######SVMClassifier
        # clf = SVC(probability=True)
        # clf.fit(X_train,y_train)
        # r = clf.score(X_test,y_test)
        # #print('SVMClassifier Mean Accuracy:%s'%r)
        # SVC_ACC.append(r)
        # predict_y_test = clf.predict(X_test)
        
        # TP=0
        # TN=0
        # FP=0
        # FN=0 
        # for i in range(0,len(y_test)):
        #     if int(y_test[i])==1 and int(predict_y_test[i])==1:
        #         TP=TP+1
        #     elif int(y_test[i])==1 and int(predict_y_test[i])==-1:
        #         FN=FN+1
        #     elif int(y_test[i])==-1 and int(predict_y_test[i])==-1:
        #         TN=TN+1
        #     elif int(y_test[i])==-1 and int(predict_y_test[i])==1:
        #         FP=FP+1
        # Sn=float(TP)/(TP+FN)
        # Sp=float(TN)/(TN+FP)
        # ACC=float((TP+TN))/(TP+TN+FP+FN)
        # SVC_Sn.append(Sn)
        # SVC_Sp.append(Sp)        
        
        # prob_predict_y_test = clf.predict_proba(X_test)
        # predictions_test = prob_predict_y_test[:, 1]
        # y_validation=np.array(y_test,dtype=int)
        # fpr, tpr, thresholds =metrics.roc_curve(y_validation, predictions_test,pos_label=1)
        # roc_auc = auc(fpr, tpr)
        # #print('SVMClassifier AUC:%s'%roc_auc)
        # SVC_AUC.append(roc_auc)
        
        # F1=metrics.f1_score(y_validation, list(map(int,predict_y_test)))
        # MCC=metrics.matthews_corrcoef(y_validation,list(map(int,predict_y_test)))
        # SVC_F1.append(F1)
        # SVC_MCC.append(MCC)
        
        # combined_prob.append(predictions_test)
        # combined_prob_array=np.array(combined_prob)
        # combined_prob_matrix=np.mat(combined_prob_array)
        # combined_prob_mean=np.mean(combined_prob_matrix,0)
        # combined_prob_mean_array=np.transpose(combined_prob_mean.getA())
        # y_validation=np.array(y_test,dtype=int)
        # fpr, tpr, thresholds =metrics.roc_curve(y_validation, combined_prob_mean_array,pos_label=1)
        # roc_auc = auc(fpr, tpr)
        
        # TP=0
        # TN=0
        # FP=0
        # FN=0 
        # for i in range(0,len(y_test)):
        #     if int(y_test[i])==1 and int(predict_y_test[i])==1:
        #         TP=TP+1
        #     elif int(y_test[i])==1 and int(predict_y_test[i])==-1:
        #         FN=FN+1
        #     elif int(y_test[i])==-1 and int(predict_y_test[i])==-1:
        #         TN=TN+1
        #     elif int(y_test[i])==-1 and int(predict_y_test[i])==1:
        #         FP=FP+1
        # Sn=float(TP)/(TP+FN)
        # Sp=float(TN)/(TN+FP)
        # ACC=float((TP+TN))/(TP+TN+FP+FN)
        # combined_Sn.append(Sn)
        # combined_Sp.append(Sp)
        # combined_ACC.append(ACC)
        # combined_AUC.append(roc_auc)         
                
        
#        plt.title('ROC Validation')
#        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
#        plt.legend(loc='lower right')
#        plt.plot([0, 1], [0, 1], 'r--')
#        plt.xlim([0, 1])
#        plt.ylim([0, 1])
#        plt.ylabel('True Positive Rate')
#        plt.xlabel('False Positive Rate')
#        plt.show()
########performance of combined classifier
#        y_validation=np.array(y_test,dtype=int)
#        fpr, tpr, thresholds =metrics.roc_curve(y_validation, combined_prob,pos_label=1)
#        roc_auc = auc(fpr, tpr)
#        combined_AUC.append(roc_auc)
    print('RandomForest Mean Accuracy:%s'%mean(RandomForest_ACC))
    print('RandomForest AUC:%s'%mean(RandomForest_AUC))
    print('RandomForest Mean Sensitive:%s'%mean(RandomForest_Sn))
    print('RandomForest Mean Specificity:%s'%mean(RandomForest_Sp))
    print('RandomForest Mean F1:%s'%mean(RandomForest_F1))
    print('RandomForest Mean MCC:%s'%mean(RandomForest_MCC))
    
    output=open('results.xls','w+',encoding='gbk')
    output.write('Acc\tAUC\tSn\tSp\tF1\tMCC\n')
    for i in range(len(RandomForest_ACC)):
        output.write(str(RandomForest_ACC[i]))
        output.write('\t')
        output.write(str(RandomForest_AUC[i]))
        output.write('\t')
        output.write(str(RandomForest_Sn[i]))
        output.write('\t')
        output.write(str(RandomForest_Sp[i]))
        output.write('\t')
        output.write(str(RandomForest_F1[i]))
        output.write('\t')
        output.write(str(RandomForest_MCC[i]))
        output.write('\t')
        output.write('\n')
    output.close()
    
#    output1=open('avg.xls','w+',encoding='gbk')
#    output1.write('Acc\tAUC\tSn\tSp\tF1\tMCC\n')
#    output1.write(str(mean(RandomForest_ACC)))
#    output1.write('\t')
#    output1.write(str(mean(RandomForest_AUC)))
#    output1.write('\t')
#    output1.write(str(mean(RandomForest_Sn)))
#    output1.write('\t')
#    output1.write(str(mean(RandomForest_Sp)))
#    output1.write('\t')
#    output1.write(str(mean(RandomForest_F1)))
#    output1.write('\t')
#    output1.write(str(mean(RandomForest_MCC)))
#    output1.write('\t')
#    output1.write('\n')
#    output1.close()
#    print('GradientBoostingClassifier Mean Accuracy:%s'%mean(GradientBoosting_ACC))
#    print('GradientBoostingClassifier AUC:%s'%mean(GradientBoosting_AUC))
#    print('GradientBoostingClassifier Mean Sensitive:%s'%mean(GradientBoosting_Sn))
#    print('GradientBoostingClassifier Mean Specificity:%s'%mean(GradientBoosting_Sp))
#    print('GradientBoostingClassifier Mean F1:%s'%mean(GradientBoosting_F1))
#    print('GradientBoostingClassifier Mean MCC:%s'%mean(GradientBoosting_MCC))         
#    
#    print('AdaBoostClassifier Mean Accuracy:%s'%mean(AdaBoost_ACC))
#    print('AdaBoostClassifier AUC:%s'%mean(AdaBoost_AUC))
#    print('AdaBoostClassifier Mean Sensitive:%s'%mean(AdaBoost_Sn))
#    print('AdaBoostClassifier Mean Specificity:%s'%mean(AdaBoost_Sp))
#    print('AdaBoostClassifier Mean F1:%s'%mean(AdaBoost_F1))
#    print('AdaBoostClassifier Mean MCC:%s'%mean(AdaBoost_MCC))
#    
#    
#    print('ExtraTreesClassifier Mean Accuracy:%s'%mean(ExtraTrees_ACC))
#    print('ExtraTreesClassifier AUC:%s'%mean(ExtraTrees_AUC))
#    print('ExtraTreesClassifier Mean Sensitive:%s'%mean(ExtraTrees_Sn))
#    print('ExtraTreesClassifier Mean Specificity:%s'%mean(ExtraTrees_Sp))
#    print('ExtraTreesClassifier Mean F1:%s'%mean(ExtraTrees_F1))
#    print('ExtraTreesClassifier Mean MCC:%s'%mean(ExtraTrees_MCC))
#        
#    
#    print('SVMClassifier Mean Accuracy:%s'%mean(SVC_ACC))
#    print('SVMClassifier AUC:%s'%mean(SVC_AUC))
#    print('SVMClassifier Mean Sensitive:%s'%mean(SVC_Sn))
#    print('SVMClassifier Mean Specificity:%s'%mean(SVC_Sp))
#    print('SVMClassifier Mean F1:%s'%mean(SVC_F1))
#    print('SVMClassifier Mean MCC:%s'%mean(SVC_MCC))
#    
#    print('combinedClassifier Mean Accuracy:%s'%mean(combined_ACC))
#    print('combinedClassifier AUC:%s'%mean(combined_AUC))
#    print('combinedClassifier Mean Sensitive:%s'%mean(combined_Sn))
#    print('combinedClassifier Mean Specificity:%s'%mean(combined_Sp))

    #print('SVMClassifier AUC:%s'%mean(combined_AUC))  
   
if __name__=='__main__':
    main(sys.argv)