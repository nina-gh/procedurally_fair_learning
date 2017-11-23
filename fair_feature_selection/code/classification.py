from __future__ import division
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import dummy
from sklearn.metrics import roc_auc_score
import pickle
from random import seed, shuffle


def generate_X_y_for_feature_set(data, features):    
    continuous_features = data["continuous_features"]  

    # generating y
    y = data["class_label"]
    y[y==0] = -1                                                         # converting class label 0 to -1

    # generating X
    X = np.array([]).reshape(len(y), 0)  
    
    for feature in features:
        feature_values = data[feature]
    
        # Preprocessing
        if feature in continuous_features:
            feature_values = [float(value) for value in feature_values]
            feature_values = preprocessing.scale(feature_values)         # 0 mean and 1 variance
            feature_values = np.reshape(feature_values, (len(y), -1))    # converting from 1-d array to a 2-d array with one column            
        else:
            one_hot = preprocessing.LabelBinarizer(neg_label=0, pos_label=1)
            feature_values = one_hot.fit_transform(feature_values)       # converting to one hot encoding
         
        X = np.hstack((X, feature_values))
		
    # if feature set is empty, return class labels
    if len(features)==0:
        feature_values = np.reshape(y, (len(y), -1))
        X = np.hstack((X, feature_values))
        
    race = data["race"]
    race = np.array([int(i) for i in race])
    
    return X, y, race


def generate_permutation(x_len):
    perm = range(0, x_len)
    seed(111)
    shuffle(perm)
    return perm
    

def calculate_classification_metrics(data, features):
    if len(features)== 0:
        clsfr = dummy.DummyClassifier(strategy="most_frequent")
    else:
        clsfr = linear_model.LogisticRegression()
    data_permutation = generate_permutation(len(data["class_label"]))
    
    # prepare data for classification
    X, y, race = generate_X_y_for_feature_set(data, features)
        
    # splitting data into train and test and preparing vector for calculating disparate mistreatment
    y = y[data_permutation]            
    X = X[data_permutation]
    race = race[data_permutation]

    total_entries = len(y)
    cut_index = int(total_entries/2.0)
    X_train, X_test = X[:cut_index], X[cut_index:]
    y_train, y_test = y[:cut_index], y[cut_index:]
    race_train, race_test = race[:cut_index], race[cut_index:]
    
    # calculating accuracy, disparate mistreatment and auc
    clsfr.fit(X_train, y_train)
    results = dict()

    predicted_labels = clsfr.predict(X_test) 
    results["predicted_labels"] = predicted_labels 
    results["accuracy"] = sum(y_test == predicted_labels)/len(y_test)
    results["auc"] = roc_auc_score(y_test, predicted_labels)
                
    # disparate mistreatment, calculating false positives and false negatives for Caucasian and non-Caucasian   
    results["fp_C"] = sum(np.logical_and(predicted_labels == 1, np.logical_and(race_test==1, y_test == -1))) /  max(sum(np.logical_and(race_test==1, y_test == -1)), 1)
    results["fn_C"] = sum(np.logical_and(predicted_labels == -1, np.logical_and(race_test==1, y_test == 1))) /  max(sum(np.logical_and(race_test==1, y_test == 1)), 1)
    results["fp_nC"] = sum(np.logical_and(predicted_labels == 1, np.logical_and(race_test==0, y_test == -1))) /  max(sum(np.logical_and(race_test==0, y_test == -1)), 1)
    results["fn_nC"] = sum(np.logical_and(predicted_labels == -1, np.logical_and(race_test==0, y_test == 1))) /  max(sum(np.logical_and(race_test==0, y_test == 1)), 1)
    results["disparate_mistreatment"] = abs(results["fp_C"] - results["fp_nC"]) + abs(results["fn_C"] - results["fn_nC"])
        
    return results

def main():
    print "classifier.py"


if __name__=="__main__":
    main()
