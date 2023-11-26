# Ablation evaluation 
from evaluation_utils import *

# Ablation evaluation


# Generate AUC MCC from the 2RNN models
# X:  Dropout rate: 0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2
# Y1: AUC value curve
# Y2: MCC value curve

from evaluation_utils import *
import pandas as pd

train_label = np.load('Ablation/model'+str(1)+'dropout_rate'+str(0.2)+'.npz')['label']

# Create an empty DataFrame to store metrics
metrics_dfRNN = pd.DataFrame(columns=['Model','MCC','AUC'])
metrics_df = pd.DataFrame(columns=['Model','AUC_train'])
for model in range(1, 3):
    for dropout_rate in [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]: 
        for activation in ['relu','tanh','sigmoid']:
            y_pred_train = np.load('Ablation/model'+str(model)+'dropout_rate'+str(dropout_rate)+'.npz')['test_predict']
            _, SN_train, SP_train, MCC_train, _, F1_train, Acc_train, AUC_train, _, _, _, _, _, y_train_pred_int = evaluation_metrics(train_label, y_pred_train)
            
            row_data = {
                'Model': f'Model {model} dropout rate {dropout_rate}',
                    'MCC': f'{MCC_train:.1%}',
                'AUC': f'{AUC_train:.1%}',
            }
            metrics_dfRNN = metrics_dfRNN.append(row_data, ignore_index=True)
            
for model in range(4, 4):
    if model != 4:
        for num_heads in [2, 4, 8, 16]:
            for dropout_rate in [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]: 
                y_pred_train = np.load('Ablation/model'+str(model)+'num_heads'+str(num_heads)+'dropout_rate'+str(dropout_rate)+'.npz')['test_predict']
                _, SN_train, SP_train, MCC_train, _, F1_train, Acc_train, AUC_train, _, _, _, _, _, y_train_pred_int = evaluation_metrics(train_label, y_pred_train)
                
                row_data = {
                    'Model': f'Model {model} num heads {num_heads} dropout rate {dropout_rate}',
                     #'MCC_train': f'{MCC_train:.1%}',
                    'AUC_train': f'{AUC_train:.1%}',
                }
                metrics_df = metrics_df.append(row_data, ignore_index=True)

                # Save plots and confusion matrix for 5CV
                #fpr, tpr, _ = roc_curve(train_label, y_pred_train)
                #plot_PR_curve(train_label, y_pred_train, f'evaluation/{model}PR_CV.png')
                #plot_ROC_curve(fpr, tpr, AUC_train, f'evaluation/{model}AUC_CV.png')
                #plot_confusion_matrix(y_true=train_label, y_pred_int=y_train_pred_int, classes=np.unique(test_label),save_path=f'evaluation/{model}confusion_CV.png')
                
# Save DataFrame to an Excel file
metrics_dfRNN.to_excel('Ablation_expRNN.xlsx', index=False)
metrics_df.to_excel('Ablation_exp.xlsx', index=False)