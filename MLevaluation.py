from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from evaluation_utils import *
import pandas as pd

# Create an empty DataFrame to store metrics
metrics_df = pd.DataFrame(columns=['Model', 'SN_train', 'SP_train', 'MCC_train', 'F1_train', 'Acc_train', 'AUC_train',
                                   'SN_test', 'SP_test', 'MCC_test', 'F1_test', 'Acc_test', 'AUC_test'])

train_label = np.load('train1.npz')['label']
test_label = np.load('test1.npz')['label']

def evaluation_ML(model_type):
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
        """
        # Calculate and save metrics to the DataFrame
        row_data = {
            'Model': f'{model_type}{model}',
            'SN_train': f'{SN_train:.1%}',
            'SP_train': f'{SP_train:.1%}',
            'MCC_train': f'{MCC_train:.1%}',
            'F1_train': f'{F1_train:.1%}',
            'Acc_train': f'{Acc_train:.1%}',
            'AUC_train': f'{AUC_train:.1%}',
            'SN_test': f'{SN_test:.1%}',
            'SP_test': f'{SP_test:.1%}',
            'MCC_test': f'{MCC_test:.1%}',
            'F1_test': f'{F1_test:.1%}',
            'Acc_test': f'{Acc_test:.1%}',
            'AUC_test': f'{AUC_test:.1%}'
        }
        metrics_df = metrics_df.append(row_data, ignore_index=True)

        # Save plots and confusion matrix for 5CV
        fpr, tpr, _ = roc_curve(train_label, y_pred_train)
        plot_ROC_curve(fpr, tpr, AUC_train, f'evaluation/{model}AUC_CV.png')
        
        # Save plots and confusion matrix for test data
        fpr, tpr, _ = roc_curve(test_label, y_pred_test)
        plot_ROC_curve(fpr, tpr, AUC_test, f'evaluation/{model}AUC_test.png')
        """
    _, SN_rf_train, SP_rf_train, MCC_rf_train, _, F1_rf_train, Acc_rf_train, AUC_rf_train, _, _, _, _, _, y_rf_train_pred_int = evaluation_metrics(train_label, fusion_train/5)
    _, SN_rf_test, SP_rf_test, MCC_rf_test, _, F1_rf_test, Acc_rf_test, AUC_rf_test, _, _, _, _, _, y_rf_test_pred_int = evaluation_metrics(test_label, fusion_test/5)

    row_data_rf = {
        'Model': f'{model_type}',
        'SN_train': f'{SN_rf_train:.1%}',
        'SP_train': f'{SP_rf_train:.1%}',
        'MCC_train': f'{MCC_rf_train:.1%}',
        'F1_train': f'{F1_rf_train:.1%}',
        'Acc_train': f'{Acc_rf_train:.1%}',
        'AUC_train': f'{AUC_rf_train:.1%}',
        'SN_test': f'{SN_rf_test:.1%}',
        'SP_test': f'{SP_rf_test:.1%}',
        'MCC_test': f'{MCC_rf_test:.1%}',
        'F1_test': f'{F1_rf_test:.1%}',
        'Acc_test': f'{Acc_rf_test:.1%}',
        'AUC_test': f'{AUC_rf_test:.1%}'
    }

    return row_data_rf
"""
for model_type in ['RF', 'svm', 'ada']:
    row = evaluation_ML(model_type)
    metrics_df = metrics_df.append(row, ignore_index=True)
metrics_df.to_excel('MLresults.xlsx', index=False)
"""
for model_type in ['svm']:
    row = evaluation_ML(model_type)
    metrics_df = metrics_df.append(row, ignore_index=True)
metrics_df.to_excel('SVMresults.xlsx', index=False)