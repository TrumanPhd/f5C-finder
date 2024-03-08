# f5C-finder
## Introduction
f5C-finder is the first neural network-based model for the identification on 5‑Formylcytidine modifications on mRNA.
This repository is the offical tensorflow vision of f5C-finder with all the codes used in the paper:  
f5C-finder: combining language model with multi-head attention for predicting 5‑Formylcytidine modifications on mRNA
![figure1](https://github.com/NWAFU-LiuLab/f5C-finder/blob/main/figure1.jpg)
## 1. Webserver with GUI
Your can use f5C-finder for inference at [our webserver](http://f5c.m6aminer.cn/)
, submitted data will be kept confidential.

![server](https://github.com/NWAFU-LiuLab/f5C-finder/assets/105758272/874aa0ed-f5b7-4d0e-b7d4-c206cdc90f39)

## 2. env
make sure your tensorflow > 2.0

## 3. Installation
You can download this Repository with
```shell
git clone https://github.com/NWAFU-LiuLab/f5C-finder.git
```
or just download with Github GUI

## 4. Source code
### 1.Ablation experiments
Ablation.py and Ablation_evaluation.py are the scripts for Ablation experiments

And for the Ablation results, please reference to the Ablation_exp.xlsx and Ablation_expRNN.xlsx

### 2.The structure of 2 LSTM-based models and 3 attention-based models

please reference: Model_lib2.py for LSTM-based models 

Fusion3feature.py, encoder1234_best.py Hash1_best.py for attention-based models

(the five encoder can be referenced from feature_selection.py utils.py and Hao_keras_toolkit.py)

### 3. model training 10-fold cross-valiation and independent test

train.py and model_evaluation.py for model training and evaluation

the results can be referenced from metrics.xlsx

### 4. RF SVM ADA the loser models VS f5C-finder

for RF ADA

can be referenced from ML.py ML_evaluation.py best_hyperparameters.xlsx

for SVM

can be referenced from SVM.py ML_evaluation.py SVMresults.xlsx

### 5. Figure

plot.py

and other files are the saved parameters from the model training.

### 6. Dataset
In the dataset, we provide the data trained and evaluated from all the models, to make it more usable we replace 'U' to 'T'，as shown in the dataset.

## Citation
It`s my pleasure if can use f5C-finder for f5C identification or improve the accuarcy on the provided f5C dataset, and please cite at 
## 

