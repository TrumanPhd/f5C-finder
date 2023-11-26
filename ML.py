from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,roc_auc_score,matthews_corrcoef   
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from utils import *
from evaluation_utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

dataset_num = 1 
k=10
classification_num = 0.5
np.random.seed(6)
x_train1,y,x_test1,y_test = F5c_onehot26(shuffle_data=False,data = dataset_num,flatten=True)
x_train2,a,x_test2,a = F5c_binary26(shuffle_data=False,data = dataset_num,flatten=True)
x_train4,a,x_test4,a = F5c_encoder(hash1,data = dataset_num)
x_train6,a,x_test6,a = F5c_encoder(condon,data = dataset_num)
x_train5,a,x_test5,a = F5c_1234(shuffle_data=False,data = dataset_num)
x_train3= np.concatenate([x_train4, x_train6, x_train5], axis=1)
x_test3 = np.concatenate([x_test4, x_test6, x_test5], axis=1)
num=len(y)
mode1=np.arange(num/2)%k
mode2=np.arange(num/2)%k
np.random.shuffle(mode1)
np.random.shuffle(mode2)
mode=np.concatenate((mode1,mode2))
test_predict_score=np.zeros(num)
# Function to evaluate the model
def evaluate_model(model, x_train, y, mode, k, feature_model, x_test, y_test, model_name):
    num = len(y)
    mode1 = np.arange(num // 2) % k
    mode2 = np.arange(num // 2) % k
    np.random.shuffle(mode1)
    np.random.shuffle(mode2)
    mode = np.concatenate((mode1, mode2))
    test_predict_score = np.zeros(num)

    for fold in range(k):
        train_label = y[mode != fold]
        test_label = y[mode == fold]
        train_feature = x_train[mode != fold]
        test_feature = x_train[mode == fold]

        model.fit(train_feature, train_label)
        test_predict_score[mode == fold] = model.predict_proba(test_feature)[:, 1]

    np.savez(f'ML/train'+model_name+str(feature_model)+'.npz', test_predict=test_predict_score, label=y)

    # Independent Test
    test_predict = model.predict_proba(x_test)[:, 1]
    np.savez(f'ML/test'+model_name+str(feature_model)+'.npz', test_predict=test_predict, label=y_test)

# Set random state
random_state = 6

# Create a DataFrame to store the best hyperparameters
hyperparameters_data = {'Feature Model': [], 'Best Parameters (RF)': [], 'Best Parameters (SVM)': [], 'Best Parameters (AdaBoost)': []}

# Loop over feature models
for feature_model in range(1, 6):
    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(eval(f'x_train{feature_model}'), y, test_size=0.1, random_state=random_state)

    # Set up RandomForestClassifier
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create RandomForestClassifier instance
    rf_model = RandomForestClassifier(random_state=random_state)

    # Use grid search for hyperparameter tuning on the validation set
    grid_search_rf = GridSearchCV(rf_model, param_grid_rf, scoring='roc_auc', cv=5)
    grid_search_rf.fit(x_val, y_val)

    # Get the best model from grid search
    best_rf_model = grid_search_rf.best_estimator_

    # Set up SVM
    param_grid_svm = {
        'C': [0.01, 0.01, 10],
        'gamma': [0.01, 0.01, 1],
        'kernel': ['linear', 'rbf', 'poly']
    }

    # Create SVM instance
    svm_model = SVC(probability=True, random_state=random_state)

    # Use grid search for hyperparameter tuning on the validation set
    grid_search_svm = GridSearchCV(svm_model, param_grid_svm, scoring='roc_auc', cv=5)
    grid_search_svm.fit(x_val, y_val)

    # Get the best model from grid search
    best_svm_model = grid_search_svm.best_estimator_

    # Set up AdaBoost
    param_grid_ada = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1],
        # Add other AdaBoost parameters as needed
    }

    # Create AdaBoostClassifier instance
    ada_model = AdaBoostClassifier(random_state=random_state)

    # Use grid search for hyperparameter tuning on the validation set
    grid_search_ada = GridSearchCV(ada_model, param_grid_ada, scoring='roc_auc', cv=5)
    grid_search_ada.fit(x_val, y_val)

    # Get the best model from grid search
    best_ada_model = grid_search_ada.best_estimator_

    # Evaluate the best models on the entire training set and independent test set
    evaluate_model(best_rf_model, eval(f'x_train{feature_model}'), y, mode, k, feature_model, eval(f'x_test{feature_model}'), y_test,model_name='rf')
    evaluate_model(best_svm_model, eval(f'x_train{feature_model}'), y, mode, k, feature_model, eval(f'x_test{feature_model}'), y_test,model_name='svm')
    evaluate_model(best_ada_model, eval(f'x_train{feature_model}'), y, mode, k, feature_model, eval(f'x_test{feature_model}'), y_test,model_name='ada')

    # Save the best hyperparameters to the DataFrame
    hyperparameters_data['Feature Model'].append(f'Model {feature_model}')
    hyperparameters_data['Best Parameters (RF)'].append(grid_search_rf.best_params_)
    hyperparameters_data['Best Parameters (SVM)'].append(grid_search_svm.best_params_)
    hyperparameters_data['Best Parameters (AdaBoost)'].append(grid_search_ada.best_params_)


# Create a DataFrame
hyperparameters_df = pd.DataFrame(hyperparameters_data)

# Save the DataFrame to an Excel file
hyperparameters_df.to_excel('best_hyperparameters.xlsx', index=False)