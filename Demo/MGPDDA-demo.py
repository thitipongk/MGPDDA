##########################################################################################################################
## Title: Meta-path based gene ontology profiles for predicting drug-disease associations
## Authors: Thitipong Kawichai, Apichat Suratanee, and Kitiporn Plaimas

## Program Name: MGPDDA-demo.py
## Program Description: - This program is to demonstrate how MGP-DDA works by using smaller data sets than those original ones.
## Input: 1) drug-disease association matrix 2) disease-function association matrix 3) drug-disease association matrix 
##        4) drug list  5) disease list  and 6) function list
## Output:
## Selected MGP-DDA compartments: - Concatenate the profile matrices of M1, M2, and M3
##                             - XGBoost's parameters: learning_rate = 0.3
##                                                     n_estimators = 500
##                                                     max_depth = 6
##                                                     min_child_weight = 3
##                             - The number of base classifiers (T) = 50, Aggregate scheme = Average
#########################################################################################################################

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
from sklearn.metrics import precision_recall_curve


def MGPDDA(drug_go_matrix, disease_go_matrix, drug_disease_matrix, drug_list, disease_list, function_list, latent_feature_percent, T, xgboost_params):
    
    #####################################################################################################################################################    
    # Step 1: Generate the matrices of meta-path based functional profiles corresponding to M1, M2, and M3 (See Algorithms 1 - 3 in Materials & Methods)
    ##################################################################################################################################################### 
    
    print("Step 1: Generating meta-path based functional profiles")
    
    total_drug = drug_list.shape[0] # The total number of drugs
    total_disease = disease_list.shape[0] # The total number of diseases
    total_function = function_list.shape[0] # The total number of functions
    
    ## Initialize the matrices of meta-path based functional profiles
    X1 = np.zeros((total_drug*total_disease, total_function), dtype = int)  # M1-based functional profile
    X2 = np.zeros((total_drug*total_disease, total_function), dtype = int)  # M2-based functional profile
    X3 = np.zeros((total_drug*total_disease, total_function), dtype = int)  # M3-based functional profile
    
    ## Labeled information for each row of the profile matrices
    drug_label, disease_label = [], []
    class_label = np.zeros(total_drug*total_disease, dtype=int)
    
    count = 0  ## the number of rows of the profile matrices that are already generated
    
    for ri in drug_list:
        for dj in disease_list:
            ## Find indices of a drug and disease
            drug_index, disease_index = np.where(drug_list == ri)[0][0], np.where(disease_list == dj)[0][0]
            
            ## Append the lists of labeled information
            drug_label.append(ri) 
            disease_label.append(dj)
            class_label[count] = int(drug_disease_matrix[drug_index, disease_index])
            
            ## Generate meta-path based functional profiles (X1, X2, and X3)
            
            ### Meta-path 1 (M1) : Drug-Function-Disease
            X1[count, :] = drug_go_matrix[drug_index, :] * disease_go_matrix[disease_index, :]
            
            ### Meta-path 2 (M2) : Drug-Function-Drug-Disease
            associated_drugs = np.where(drug_disease_matrix[:, disease_index] != 0)[0]  # R_dj in Algorithm 2
            total_associated_drugs = len(associated_drugs)  # The total number of the dj-associated drugs
            if total_associated_drugs != 0:
                for u in range(total_associated_drugs):
                    X2[count, :] = X2[count, :] + (drug_go_matrix[drug_index, :] * drug_go_matrix[associated_drugs[u], :])
            
            ### Meta-path 3 (M3) : Drug-Disease-Function-Disease
            associated_diseases = np.where(drug_disease_matrix[drug_index, :] != 0)[0]  # D_ri in Algorithm 3
            total_associated_diseases = len(associated_diseases)  # The total number of the ri-associated diseases
            if total_associated_diseases != 0:
                for v in range(total_associated_diseases):
                    X3[count, :] = X3[count, :] + (disease_go_matrix[disease_index, :] * disease_go_matrix[associated_diseases[v], :])
            
            ## Count the number of profile rows that are already computed
            count = count + 1  
    
    #####################################################################################################################################################    
    # Step 2: Reduce dimension of the concatenated meta-path based functional profiles using SVD
    ##################################################################################################################################################### 
    
    print("Step 2: Reducing dimension of the concatenated profile matrix using SVD")
    
    ## Concatenate the profile matrices of M1, M2, and M3 
    concatenated_profile = sparse.csr_matrix(np.hstack((X1, X2, X3)))
    
    ## Extract features of the concatenated profile matrix using SVD
    num_feature = concatenated_profile.shape[1] # The total number of the original features
    latent_feature_num = int(num_feature * latent_feature_percent/100) # The number of latent features required
    svd = TruncatedSVD(n_components=latent_feature_num, n_iter=7, random_state=16)
    svd.fit(concatenated_profile)
    transformed_profile = svd.transform(concatenated_profile)

    
    #####################################################################################################################################################    
    # Step 3: Train MGP-DDA and predict all unlabeled drug-disease pairs
    ##################################################################################################################################################### 
    
    print("Step 3: Train MGP-DDA model and predict all unlabeled drug-disease pairs to identify the potential ones") 
    
    ## Find indices of positive and unlabeled drug-disease pairs according to the transformed profile matrix
    positive_row = np.where(class_label == 1)[0]
    unlabeled_row = np.where(class_label == 0)[0]
    
    #total_unlabeled = unlabeled_row.shape[0] # The total number of unlabeled pairs
    
    ## Determine the feature matrix of all unlabeled drug-disease pairs for making predictions
    #unlabeled_feature_matrix = transformed_profile[unlabeled_row, :]
    
    ## Initialize the vector of predicted scores for all unlabeled drug-disease pairs
    sum_predicted_score = np.zeros(total_drug*total_disease, dtype=float)
    
    ## Train each base classifier (XGBoost) & predict all unlabeled drug-disease pairs
    for t in range(T):
        #print("-> Running the base classifier no." + str(t+1) + "/" + str(T))
        ## Create a bootstrap sample of unlabeled samples with the same size of the positive ones
        bootstrap_unlabeled_row = np.random.RandomState(t).choice(unlabeled_row, size = positive_row.shape[0], replace = True)
        ## Create the training feature matrix
        train_row = np.append(positive_row, bootstrap_unlabeled_row)
        train_feature_matrix = transformed_profile[train_row, :]
        train_label_vector = class_label[train_row]
        
        ## Train XGBoost classifier no.t, where t = 0, 1, ..., T-1
        ### Assign values of XGBoost parameters
        p1, p2, p3, p4 = xgboost_params['learning_rate'], int(xgboost_params['n_estimators']), int(xgboost_params['max_depth']), int(xgboost_params['min_child_weight']) 
        ### Train a base classifier (XGBoost)
        clf = xgb.XGBClassifier(random_state = 1, learning_rate = p1, n_estimators = p2, max_depth = p3, min_child_weight = p4, n_jobs=-1)
        clf.fit(train_feature_matrix, train_label_vector)
        ## Collect an aggregate predictions obtained from the trained base classifier
        sum_predicted_score = sum_predicted_score + clf.predict_proba(transformed_profile)[:, 1]
    
    ## Calculate the average scores of the predictions
    avg_predicted_score = sum_predicted_score/T
    
    ## Compute a binary class for each unlabeled drug-disease pairs using a threshold score awarding the maximum F1, PU
    ### Retrieve Precision, Recall for each threshold score of the Precision-Recall curve
    precision, recall, thresholds_pr = precision_recall_curve(class_label, avg_predicted_score)
    ### Calculate the probability for positive predicting at each threshold_pr score (See Eq 8 in Materials & methods)
    prob_pred_pos = np.zeros((len(thresholds_pr)), dtype=float)  # Estimated probability of positive predictions
    for i in range(len(thresholds_pr)):
        prob_pred_pos[i] = len(np.where(avg_predicted_score >= thresholds_pr[i])[0])/len(avg_predicted_score)
    ### Find a threshold score giving the maximum F1,PU
    F1_pu = (recall[range(len(thresholds_pr)-1)]**2)/prob_pred_pos[range(len(thresholds_pr)-1)]
    max_index = np.argmax(F1_pu)
    threshold_score = thresholds_pr[max_index]
    ### Predict binary classes
    predicted_class = np.copy(avg_predicted_score)
    predicted_class[predicted_class >= threshold_score] = 1
    predicted_class[predicted_class < threshold_score] = 0
    predicted_class = predicted_class.astype(int)
    
    ## Write the aggregate predictions of all unlabeled drug-disease pairs
    selected_drug_label = [drug_label[i] for i in unlabeled_row]
    selected_disease_label = [disease_label[i] for i in unlabeled_row]
    result_df = pd.DataFrame({'drugbankId': selected_drug_label, 'omimId': selected_disease_label, 'predictedScore': avg_predicted_score[unlabeled_row], 'predictedClass': predicted_class[unlabeled_row]})
    result_df.to_csv('output-Predictions of all unlabeled pairs.csv', index = False)
    
    print("")
    print("Completed: Results are in the file named 'output-Predictions of all unlabeled pairs.csv'. ")
    

if __name__ == "__main__":
    
    print("")
    print("=========================================================================================================")
    print("Demo version of Meta-path based Gene ontology Profiles for predicting Drug-Disease Associations (MGP-DDA)")
    print("=========================================================================================================")
    print("")
    
    # Set values to MGP-DDA parameters
    latent_feature_percent = 1.28  # The percentage of required latent features for using the meta-paths M1, M2, and M3 (See Table S1 in S1 Appendix)
    T = 50  # The number of base classifiers
    xgboost_params = {'learning_rate': 0.3, 'n_estimators': 500, 'max_depth': 6, 'min_child_weight': 3}
    
    # Read drug-function, disease-function, and drug-disease association matrices
    drug_go_matrix = np.loadtxt('./Datasets-demo/drug-function_association_matrix-demo.txt', delimiter='\t', dtype=int)
    disease_go_matrix = np.loadtxt('./Datasets-demo/disease-function_association_matrix-demo.txt', delimiter='\t', dtype=int)
    drug_disease_matrix = np.loadtxt('./Datasets-demo/drug-disease_association_matrix-demo.txt', delimiter='\t', dtype=int)
    
    # Read drug and diseaselist
    drug_list = (pd.read_csv('./Datasets-demo/drug_list-demo.csv'))['drugBankId']
    disease_list = (pd.read_csv('./Datasets-demo/disease_list-demo.csv'))['omimId']
    function_list = (pd.read_csv('./Datasets-demo/function_list-demo.csv'))['GoId']
    
    MGPDDA(drug_go_matrix, disease_go_matrix, drug_disease_matrix, drug_list, disease_list, function_list, latent_feature_percent, T, xgboost_params)
    
    