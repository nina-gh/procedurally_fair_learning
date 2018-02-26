from __future__ import division
import numpy as np
import pickle
from collections import defaultdict, Counter
from itertools import chain, combinations
from random import seed, shuffle
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import roc_auc_score

from preprocess_classification_datasets import load_preprocessed_classification_data, preprocess_all_datasets


DATA_FOLDER = "../data/"


#######################
## utility functions ##
#######################

def generate_permutation(random_seed, x_len):
    perm = range(0, x_len)
    seed(random_seed)
    shuffle(perm)
    return perm

##################################
## prep data for classification ##
##################################

def generate_X_y(data, features, control_features):    
    continuous_features = data["continuous_features"]  

    # generating y
    y = data["class_label"]
    y[y==0] = -1    # converting class label 0 to -1

    # generating X
    X = np.array([]).reshape(len(y), 0) 

    # generating control
    control = dict()
    for control_feature in control_features: 
        control[control_feature] = data[control_feature]

    # preprocessing
    for feature in features:
        feature_values = data[feature]
        if feature in continuous_features:
            feature_values = [float(value) for value in feature_values]
            feature_values = preprocessing.scale(feature_values)         # 0 mean and 1 variance
            feature_values = np.reshape(feature_values, (len(y), -1))    # converting from 1-d array to a 2-d array with one column            
        else:
            one_hot = preprocessing.LabelBinarizer(neg_label=0, pos_label=1)
            feature_values = one_hot.fit_transform(feature_values)       # converting to one hot encoding
        
        X = np.hstack((X, feature_values))

    # if feature set is empty, return class labels as X
    if len(features)==0:
        feature_values = np.reshape(y, (len(y), -1))
        X = np.hstack((X, feature_values))

    return X, y, control


def prep_train_test(data, features, control_features, frac_train=0.5, random_seed=111):
    # generate X, y and control vectors
    X, y, control_dict = generate_X_y(data, features, control_features)
    worker_control = control_dict["worker"]
    fairness_control = control_dict["fairness"]
    
    # shuffle data by creating a shuffled array of workers
    data_permutation = generate_permutation(random_seed, len(set(worker_control)))
    sorted_workers = np.array(sorted(set(worker_control)))
    permuted_workers = sorted_workers[data_permutation]

    # find split indices from shuffled array of workers
    cut_index = int(len(data_permutation) * frac_train)        
    train_workers, test_workers = permuted_workers[:cut_index], permuted_workers[cut_index:]
    train_indices, test_indices = np.in1d(data["worker"], train_workers), np.in1d(data["worker"], test_workers)

    # splitting data into train and test
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    fairness_control_train, fairness_control_test = fairness_control[train_indices], fairness_control[test_indices]
    worker_control_train, worker_control_test = worker_control[train_indices], worker_control[test_indices]

    return X_train, X_test, y_train, y_test, fairness_control_train, fairness_control_test, worker_control_train, worker_control_test


######################
## train classifier ##
######################

def train_classifier(data, features, control_features, frac_train=0.5, random_seed=111):
    # generate train and test data
    X_train, X_test, y_train, y_test, fairness_control_train, fairness_control_test, worker_control_train, worker_control_test = prep_train_test(data, features, control_features, frac_train=frac_train, random_seed=random_seed)

    # train model
    clsfr = linear_model.LogisticRegression()
    clsfr.fit(X_train, y_train)

    return clsfr


######################
## make predictions ##
######################

def make_predictions(clsfr, data, data_subset, features, control_features, frac_train=0.5, random_seed=111):
    # generate train and test data
    X_data, y_data, fairness_control_data, worker_control_data = dict(), dict(), dict(), dict()

    X_data["train"], X_data["test"], y_data["train"], y_data["test"], fairness_control_data["train"], fairness_control_data["test"], worker_control_data["train"], worker_control_data["test"] = prep_train_test(data, features, control_features, frac_train=frac_train, random_seed=random_seed)

    X_data["all"], y_data["all"], control_dict = generate_X_y(data, features, control_features)
    fairness_control_data["all"] = control_dict["fairness"]
    worker_control_data["all"] = control_dict["worker"]

    # choose evaluation data
    X = X_data[data_subset]
    y = y_data[data_subset]
    fairness_control = fairness_control_data[data_subset]
    worker_control = worker_control_data[data_subset]

    # make predictions
    ground_truth = y
    predicted = clsfr.predict(X)
    predicted_prob = clsfr.predict_proba(X)

    return ground_truth, predicted, predicted_prob, fairness_control, worker_control


#########################
## evaluate classifier ##
#########################

def calculate_evaluation_metrics(ground_truth, predicted):
    accuracy = sum(ground_truth == predicted)/len(predicted)
    auc = roc_auc_score(ground_truth, predicted)

    return accuracy, auc


def characterize_mistakes_per_rating(ground_truth, predicted, predicted_prob, fairness_control):
    # indices of incorrect predictions
    mistakes = predicted != ground_truth
    
    # incorrect predictions per fairness rating
    dist_rating = Counter(fairness_control)
    mistakes_per_rating = Counter(fairness_control[mistakes])
    frac_mistakes_per_rating = {key: mistakes_per_rating[key]/dist_rating[key] for key in mistakes_per_rating}

    results = dict()
    for fairness_rating in dist_rating:
        results[fairness_rating] = dict()
        row_indices = fairness_control == fairness_rating

        # total instances
        results[fairness_rating]['num_instances'] = np.sum(row_indices)

        # frac misclassified
        frac_misclassified = (predicted[row_indices] != ground_truth[row_indices]).mean()
        results[fairness_rating]['frac_misclassified'] = round(frac_misclassified, 2)

        # avg distance from boundary
        column_indices = [i if i == 1 else 0 for i in ground_truth[row_indices]]
        relevant_pp = predicted_prob[row_indices, column_indices]    
        results[fairness_rating]['avg_prob'] = round(relevant_pp.mean(), 2)
        results[fairness_rating]['std_prob'] = round(np.std(relevant_pp), 2)
    
    return results


def characterize_mistakes_per_worker(ground_truth, predicted, worker_control):
    # indices of correct predictions
    correct_predictions = predicted == ground_truth
    
    # correct predictions per worker
    dist_worker = Counter(worker_control)
    correct_per_worker = Counter(worker_control[correct_predictions])
    frac_correct_per_worker = {key: correct_per_worker[key]/dist_worker[key] for key in correct_per_worker}

    # prep cdf
    worker_cdf = list()

    pre_worker_cdf = Counter(frac_correct_per_worker.values())
    thresholds = [t/10 for t in range(0,11)]
    for t in thresholds:
        worker_cdf.append( sum( [pre_worker_cdf[key] for key in pre_worker_cdf if key<= t] ) )

    worker_cdf = [i/len(dist_worker) for i in worker_cdf]

    return zip(thresholds, worker_cdf)
