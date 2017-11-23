from __future__ import division
import os, sys
import numpy as np
from collections import defaultdict
from classification import calculate_classification_metrics
import pickle

DATA_FOLDER = "../data/"

def feature_increases_acc_null_set(folder, feature, epsilon):
    count = dict()
    data = pickle.load(open(DATA_FOLDER + "ProPublica_clf_data.pickle"))

    feature_set_wo = []
    feature_set_w = [feature]
                
    acc_w = calculate_classification_metrics(data, feature_set_w)["accuracy"]
    acc_wo = calculate_classification_metrics(data, feature_set_wo)["accuracy"]
    if acc_w - acc_wo >= epsilon:
        count["inc_accuracy"]=1
    else:
        count["inc_accuracy"]=0
            
    return count
    
    
def feature_increases_disp_null_set(folder, feature, epsilon):
    count = dict()
    data = pickle.load(open(DATA_FOLDER + "ProPublica_clf_data.pickle"))
    
    feature_set_wo = []
    feature_set_w = [feature]
            
    disp_w = calculate_classification_metrics(data, feature_set_w)["disparate_mistreatment"]
    disp_wo = calculate_classification_metrics(data, feature_set_wo)["disparate_mistreatment"]
    if disp_w - disp_wo >= epsilon:
        count["inc_disparity"]=1
    else:
        count["inc_disparity"]=0 
            
    return count


def ProPublica_increase_decrease_disp_acc():
    # epsilon_A = 0.05 * (accuracy(all features) - accuracy(empty set))
    # epsilon_D = 0.05 * (disparate_mistreatment(all features) - disparate_mistreatment(empty set))
    epsilon_A = 0.006
    epsilon_D = 0.013
    clsfr_data = pickle.load(open(DATA_FOLDER + "ProPublica_clf_data.pickle"))
    classification_features = clsfr_data["classification_features"]

    data = defaultdict(dict)
        
    for feature in classification_features:
        data["increase_decrease_accuracy"][feature] = feature_increases_acc_null_set(clsfr_data, feature, epsilon_A)
        data["increase_decrease_disparity"][feature] = feature_increases_disp_null_set(clsfr_data, feature, epsilon_D)

    pickle.dump(data, open(DATA_FOLDER + "ProPublica_increase_decrease_disp_acc_null_set.pickle", "w"))    
    
    
def main():
    ProPublica_increase_decrease_disp_acc()

    
if __name__=="__main__":
    main()