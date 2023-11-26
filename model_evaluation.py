from evaluation_utils import *
import pandas as pd

train_label = np.load('train1.npz')['label']
test_label = np.load('test1.npz')['label']

# Create an empty DataFrame to store metrics
metrics_df = pd.DataFrame(columns=['Model', 'SN_train', 'SP_train', 'MCC_train', 'F1_train', 'Acc_train', 'AUC_train',
                                   'SN_test', 'SP_test', 'MCC_test', 'F1_test', 'Acc_test', 'AUC_test'])

fusion_train = np.zeros(2888)
fusion_test = np.zeros(722)

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
    
    # Calculate and save metrics to the DataFrame
    row_data = {
        'Model': f'Model {model}',
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

fusion_train = fusion_train/5
fusion_test =  fusion_test/5

_, SN_train, SP_train, MCC_train, _, F1_train, Acc_train, AUC_train, _, _, _, _, _, y_train_pred_int = evaluation_metrics(train_label, fusion_train)
_, SN_test, SP_test, MCC_test, _, F1_test, Acc_test, AUC_test, _, _, _, _, _, y_test_pred_int = evaluation_metrics(test_label, fusion_test)

row_data = {
    'Model': 'fusion',
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


# Save DataFrame to an Excel file
metrics_df.to_excel('metrics.xlsx', index=False)
