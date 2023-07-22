# f5C-finder
## 1. Introduction
f5C-finder is the first neural network-based model for the identification on 5‑Formylcytidine modifications on mRNA.
This repository is the offical tensorflow vision of f5C-finder with all the codes used in the paper:  
f5C-finder: combining language model with multi-head attention for predicting 5‑Formylcytidine modifications on mRNA
## Webserver with GUI
Your can use f5C-finder for inference at [our webserver](http://f5c.m6aminer.cn/)
, submitted data will be kept confidential.

## 2. env
make sure your tensorflow > 2.0

## 3. Installation
You can download this Repository with
```shell
git clone https://github.com/NWAFU-LiuLab/f5C-finder.git
```
or just download with Github GUI

## 4. Source code
### statisticallearning/

(the source code used in the feature subset selection)

classifier_selection.py include the 6 statistical classifier used in this paper

feature_selection.py is the lib with different feature extraction methods

### deeplearning/

Moddel_lib2.py include the main network-based models\

main_final.py is the final test of f5C_finder

and other python files are used for the results display


## 5. Train on your own dataset
### You can use our webserver GUI for inference, or train with your own dataset for f5C identification as follow:

format dataset (the input dataset should be in .txt formation)

the length of the sequences should must be 101

### change the path and add
```python 
model.predict(seq_to_predict)
```

## 6. Citation
It`s my pleasure if can use f5C-finder for f5C identification or improve the accuarcy on the provided f5C dataset, and please cite at 
## 

