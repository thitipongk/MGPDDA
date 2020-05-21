# PUMDR

This repository contains the code and data sets of the Positive-unlabeled learning method with Meta-path based functional profiles for Drug Repositioning (PUMDR).

## 1. Requirements
The code is written in Python (version > 3.6). The following packages are required to run the code:
- numpy (for matrix manipulation and computation)
- pandas (for reading input files in the format of csv)
- scipy (for converting profile matrices to sparse matrices)
- sklearn (for computing Precision and Recall)
- xgboost (for learning and predicting drug-disease pairs).

## 2. Demo
This folder contains the code of PUMDR (PUMDR-demo.py) and the exemplified data to demonstrate how to PUMDR works.
### 2.1 PUMDR-demo code
This code is for demonstrate how to PUMDR works with the following processes:
- Generation of meta-path based functional profiles 
- Dimensionality reduction of the functional profiles
- Training the PUMDR model and predicting all unlabeled pairs to identify potential drug-disease associations. 
Note that this code may not suitable for our original data sets due to their too large size. Anyway, we also provided an exemplified data sets for this code.

### 2.2 Datasets-demo

## 3. Datasets

## 4. Citation
