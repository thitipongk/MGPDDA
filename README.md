# PUMDR

This repository contains the code and data sets of the Positive-Unlabeled learning method with Meta-path based functional profiles for Drug Repositioning (PUMDR). The folder "Demo" contains the code of PUMDR and the exemplified data. Another folder "Datasets" contains all original data sets used in our experiments.

## 1. Requirements
The code is written in Python (version > 3.6). The following packages are required to run the code:
- numpy (for matrix manipulation and computation)
- pandas (for reading input files in the format of csv)
- scipy (for manipulating sparse profile matrices)
- sklearn (for computing Precision and Recall)
- xgboost (for learning and predicting drug-disease pairs).

## 2. Demo
This folder contains the code of PUMDR (PUMDR-demo.py) and the exemplified data to demonstrate how PUMDR works.
### 2.1 PUMDR-demo code
This code is for demonstrate how PUMDR works, composing of these following processes:
1) Generation of meta-path based functional profiles 
2) Dimensionality reduction of the functional profiles
3) Training the PUMDR model and predicting all unlabeled pairs to identify potential drug-disease associations. 

Note that this code may not be suitable for our original data due to their too large size. Anyway, we also provided the exemplified data for this code to illustrate all processes of PUMDR.

### 2.2 Datasets-demo
In the exemplified data set, 100 drugs and 50 diseases were randomly selected from the original data. The total number of the associated GO functions is 4,669. The descriptions of all input data are shown as follows:
1) "drug_list-demo.csv" : the list of 100 drugs (DrugBank ID)
2) "disease_list-demo.csv" : the list of 50 diseases (OMIM ID)
3) "function_list-demo.csv" : the list of 4,669 GO functions (GO ID) and their aspects, including Cellular Component (C), Molecular Function (F), and Biological Process (P).
4) "drug-function_association_matrix-demo.txt" : the matrix of drug-function associations (tab-delimited)
5) "disease-function_association_matrix-demo.txt" : the matrix of disease-function associations (tab-delimited)
6) "drug-disease_association_matrix-demo.txt" : the matrix of drug-disease associations (tab-delimited)
Note that the orders of the data in the association matrices are arranged in the same orders of those in the listing files.

## 3. Datasets
This folder contains the original data that we used in the experiments. These are quite large data, containing 1,022 drugs, 585 diseases, and 8,320 GO functions. We also provided the matrices of the pre-constructed functional profiles based on each meta-paths for the users who would like to reproduce our prediction results. The descriptions of the all data are shown as follows:
1) "drug_list.csv" : the list of 1,022 drugs (DrugBank ID)
2) "disease_list.csv" : the list of 585 diseases (OMIM ID)
3) "function_list.csv" : the list of 8,320 GO functions (GO ID) with their aspects, including Cellular Component (C), Molecular Function (F), and Biological Process (P).
4) "drug-function_association_matrix.txt" : the matrix of drug-function associations (tab-delimited)
5) "disease-function_association_matrix.txt" : the matrix of disease-function associations (tab-delimited)
6) "drug-disease_association_matrix.txt" : the matrix of drug-disease associations (tab-delimited)
7) "full_M1-based_functional_profiles.npz" : the full functional profiles based on the meta-path M<sub>1</sub> (no column and row names)
8) "full_M2-based_functional_profiles.npz" : the full functional profiles based on the meta-path M<sub>2</sub> (no column and row names)
9) "full_M3-based_functional_profiles.npz" : the full functional profiles based on the meta-path M<sub>3</sub> (no column and row names)
10) "all_drug-disease_pairs.csv" : the list of all drug-disease pairs with their class labels, one is positive and zero is unlabeled.

Note that the files no. 7 - 9 are the Python sparse matrices in the format of Compressed Sparse Row (CSR). We recommend to import the module "sparse" from the scipy package to read these files. Although these files have no row names, you can match each row of the functional profiles to each drug-disease pair in the file no.10. For the column names of the functional profiles, you can match each column to each function in the file no.3. The orders of the data in the association and profile matrices are arranged in the same orders of those in the listing files.

## 4. Citation