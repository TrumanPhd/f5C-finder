from evaluation_utils import *
import pandas as pd

train_label = np.load('train1.npz')['label']
test_label = np.load('test1.npz')['label']

# Create an empty DataFrame to store metrics
metrics_df = pd.DataFrame(columns=['Model', 'SN_train', 'SP_train', 'MCC_train', 'F1_train', 'Acc_train', 'AUC_train',
                                   'SN_test', 'SP_test', 'MCC_test', 'F1_test', 'Acc_test', 'AUC_test'])

fusion_train = np.zeros(2888)
fusion_test = np.zeros(722)



def ML(model_type):
    # Loop over models
    fusion_train = np.zeros(2888)
    fusion_test = np.zeros(722) 
    for model in range(1, 6):
        y_pred_train = np.load(f'ML/train{model_type}{model}.npz')['test_predict']
        _, SN_train, SP_train, MCC_train, _, F1_train, Acc_train, AUC_train, _, _, _, _, _, y_train_pred_int = evaluation_metrics(train_label, y_pred_train)
        fusion_train += y_pred_train
        y_pred_test = np.load(f'ML/test{model_type}{model}.npz')['test_predict']  
        _, SN_test, SP_test, MCC_test, _, F1_test, Acc_test, AUC_test, _, _, _, _, _, y_test_pred_int = evaluation_metrics(test_label, y_pred_test)
        fusion_test += y_pred_test
    return fusion_train/5, fusion_test/5


for model in range(1, 6):
    # Training data evaluation
    y_pred_train = np.load(f'train{model}.npz')['test_predict']
    _, SN_train, SP_train, MCC_train, _, F1_train, Acc_train, AUC_train, _, _, _, _, _, y_train_pred_int = evaluation_metrics(train_label, y_pred_train)
    
    # if model == 1:
    fusion_train += y_pred_train
    
    # Test data evaluation
    y_pred_test = np.load(f'test{model}.npz')['test_predict']  
    _, SN_test, SP_test, MCC_test, _, F1_test, Acc_test, AUC_test, _, _, _, _, _, y_test_pred_int = evaluation_metrics(test_label, y_pred_test)
    
    # if model == 1:
    fusion_test += y_pred_test
    
fusion_train = fusion_train/5
fusion_test =  fusion_test/5

_, SN_train, SP_train, MCC_train, _, F1_train, Acc_train, AUC_train, _, _, _, _, _, fusion_train_int = evaluation_metrics(train_label, fusion_train)
_, SN_test, SP_test, MCC_test, _, F1_test, Acc_test, AUC_test, _, _, _, _, _,fusion_test_int = evaluation_metrics(test_label, fusion_test)

# Save plots and confusion matrix for 5CV
fpr, tpr, _ = roc_curve(train_label, fusion_train)

plot_PR_curve(train_label, fusion_train_int, f'evaluation/fusion_PR_CV.png')
plot_ROC_curve(fpr, tpr, AUC_train, f'evaluation/fusion_AUC_CV.png')
#plot_confusion_matrix(y_true=train_label, y_pred_int=y_train_pred_int, classes=np.unique(test_label),save_path=f'evaluation/fusion_confusion_CV.png')

dict_name = {'RF':'Fusion-RF','svm':'Fusion-SVM','ada':'Fusion-ADA'}
def curve_generator2(label,y_pred,model_type,th=0.5):
    y_predlabel = [(0 if item < th else 1) for item in y_pred]
    y_predlabel = np.array(y_predlabel)
    return (label,y_pred,dict_name[model_type])




curves = []
fpr, tpr, _ = roc_curve(train_label, fusion_train)
curves.append((fpr,tpr,AUC_train,'f5C-finder'))
for model_type in ['RF','svm','ada']:  
    y_pred,_ = ML(model_type)
    curve = curve_generator(train_label, y_pred,model_type)
    curves.append(curve)
MultiROC(curves,save_path='AUCtrain.tif')

curves = []
fpr, tpr, _ = roc_curve(test_label,fusion_test)
curves.append((fpr,tpr,AUC_test,'f5C-finder'))
for model_type in ['RF','svm','ada']:  
    _,y_pred = ML(model_type)
    curve = curve_generator(test_label, y_pred,model_type)
    curves.append(curve)
MultiROC(curves,save_path='AUCtest.tif')




curves = []
curves.append((train_label,fusion_train,'f5C-finder'))
for model_type in ['RF','svm','ada']:  
    y_pred,_ = ML(model_type)
    curve = curve_generator2(train_label,y_pred,model_type)
    curves.append(curve)
MultiPR(curves,save_path='PRtrain.tif')

curves = []
curves.append((test_label,fusion_test,'f5C-finder'))
for model_type in ['RF','svm','ada']:  
    _,y_pred = ML(model_type)
    curve = curve_generator2(test_label, y_pred,model_type)
    curves.append(curve)
MultiPR(curves,save_path='PRtest.tif')

