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
from classification import train_classifier, make_predictions, calculate_evaluation_metrics, characterize_mistakes_per_rating, characterize_mistakes_per_worker
from consensus import calculate_concensus


DATA_FOLDER = "../data/"


def print_table(results):
    table_text = ""
    table_text += "Rating"
    for key in sorted(results):
        table_text += " & {}".format(key)
    table_text += "\\\\\n"

    titles = ["Total num", "Fraction Misclassified", "Avg Prob Correct Class", "Std Of Prob Correct Class"]
    keys = ["num_instances", "frac_misclassified", "avg_prob", "std_prob"]

    for i, title in enumerate(titles):
        table_text += "{}".format(title)
        for key in sorted(results):
            table_text += " & {}".format(results[key][keys[i]])
        table_text += "\\\\\n"

    print table_text


def print_cdf(results):
    for i, j in results:
        print "{}, {}".format(i, round(j, 2))


# uredi ispisivanje da radi i za verziju bez neutral s_ili_bez_neutral_ratingsima   30 min


def demo():        
    frac_train = 0.5
    random_seed = 560
    dataset_prefix = "AMT" # choose between "AMT", "AMT_wo_neutral", "SSI", "SSI_wo_neutral"
    data_subset = "test" # choose between "all", "train", "test"
    features = ['volitionality', 'reliability', 'privacy', 'relevance', 'causes_outcome', 'caused_by_sensitive_feature', 'causal_loop', 'causes_disparity_in_outcomes']
    control_features = ["fairness", "worker"]

    # load preprocessed dataset
    preprocess_all_datasets()
    data = load_preprocessed_classification_data(dataset_prefix, show_preview=False)


    # CLASSIFICATION
    # train & evaluate classifiers
    accuracy_cv, auc_cv = list(), list()
    for i in range(0, 5):
        random_seed -= 1
        # train classifier
        clsfr = train_classifier(data, features, control_features, frac_train=frac_train, random_seed=random_seed)    
        # make predictions
        ground_truth, predicted, predicted_prob, fairness_control, worker_control = make_predictions(clsfr, data, "test", features, control_features, frac_train=frac_train, random_seed=random_seed)    
        ## accuracy & auc
        accuracy, auc = calculate_evaluation_metrics(ground_truth, predicted)
        accuracy_cv.append(accuracy)
        auc_cv.append(auc)

    print "Average accuracy: ", np.average(np.array(accuracy_cv))
    print "Average AUC: ", np.average(np.array(auc_cv))

    # characterize misclassifications
    ## evaluate on whole data
    ground_truth, predicted, predicted_prob, fairness_control, worker_control = make_predictions(clsfr, data, "all", features, control_features, frac_train=frac_train, random_seed=random_seed)    
    
    ## missclassifications per fairness rating
    rating_mistakes = characterize_mistakes_per_rating(ground_truth, predicted, predicted_prob, fairness_control)
    print "\n\nCharacterize misclassifications per fairness rating\n"
    print_table(rating_mistakes)

    ## misclassifications per worker
    print "\n\nCharacterize misclassifications per worker (CDF)\n"
    worker_mistakes_cdf = characterize_mistakes_per_worker(ground_truth, predicted, worker_control)
    print_cdf(worker_mistakes_cdf)


    # CONSENSUS
    clsfr = train_classifier(data, features, control_features, frac_train=frac_train, random_seed=555)
    ground_truth, predicted, predicted_prob, fairness_control, worker_control = make_predictions(clsfr, data, "all", features, control_features, frac_train=frac_train, random_seed=random_seed)    
    data["predicted_fairness"] = predicted
    concensus = calculate_concensus(data)




def main():
    demo()
         
    
if __name__=="__main__":
    main()
