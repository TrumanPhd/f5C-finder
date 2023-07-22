# f5C-finder
## Introduction
f5C-finder is the first neural network-based model for the identification on 5‑Formylcytidine modifications on mRNA.
This repository is the offical tensorflow vision of f5C-finder with all the codes used in the paper:  
f5C-finder: combining language model with multi-head attention for predicting 5‑Formylcytidine modifications on mRNA
## Webserver with GUI
Your can use f5C-finder for inference at [our webserver](http://f5c.m6aminer.cn/)
, submitted data will be kept confidential.

## env
make sure your tensorflow > 2.0

## Installation
You can download this Repository with
```shell
git clone https://github.com/NWAFU-LiuLab/f5C-finder.git
```
or just download with Github GUI

## Train on your own dataset
### You can use our webserver GUI for inference, or train with your own dataset for f5C identification as follow:

format dataset (the input dataset should be in .txt formation)

the length of the sequences should must be 101

### change the path and add
```python 
model.predict(seq_to_predict)
```

## Citation
It`s my pleasure if can use f5C-finder for f5C identification or improve the accuarcy on the provided f5C dataset, and please cite at 
## 

