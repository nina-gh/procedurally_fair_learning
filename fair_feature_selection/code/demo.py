from __future__ import division
import numpy as np
import pandas as pd
import pickle
from submodular_optimization import submodular_optimization

DATA_FOLDER = "../data/"

def load_fairness_data(data_path_amt, data_path_inc_dec, unfairness_type):
    df = pd.read_csv(data_path_amt)
    data = df.to_dict('list')
    for k in data.keys():
        data[k] = np.array(data[k])
        
    data["fairness_definition"] = unfairness_type
    increase_decrease_data = pickle.load(open(data_path_inc_dec))
    data["increase_decrease_accuracy"] = increase_decrease_data["increase_decrease_accuracy"]
    data["increase_decrease_disparity"] = increase_decrease_data["increase_decrease_disparity"]

    return data


def load_accuracy_data(data_path):
    data = pickle.load(open(data_path))
    return data


def calculate_ragnes(min_val, max_val):
    diff = max_val - min_val    
    ranges = list()
    
    cs = np.arange(0, 1, 0.1)
    for c in cs:
        val = round((min_val + diff*c),3)
        ranges.append(val)
    
    return ranges


def demo():
    unfairness_data = load_fairness_data(DATA_FOLDER + "ProPublica_amt_data.csv", DATA_FOLDER + "ProPublica_increase_decrease_disp_acc_null_set.pickle","apriori")
    accuracy_data = load_accuracy_data(DATA_FOLDER + "ProPublica_clf_data.pickle")
    
    range_ISSC = calculate_ragnes(0.54, 0.68)
    range_ISK = calculate_ragnes(0.0, 1)
    
    # minimizing unfairness s.t. a constraint on accuracy
    print "*** Minimizing unfairness s.t. a constraint on accuracy ***"
    for t in range_ISSC:
        algorithm = "SSC"
        res = submodular_optimization(unfairness_data, accuracy_data, t, algorithm)
        print "Minimize unfairness s.t. accuracy >= {}".format(t)
        if res[0] is None:
            print "No feature set satisfies these constraints."
        else:
            print "Feature set: {}, accuracy = {}, unfairness = {}\n".format(",".join(res[0]), round(res[2],2), round(res[3],2))
        
    # maximizing accuracy s.t. a constraint on unfairness
    print "\n\n*** Maximizing accuracy s.t. a constraint on unfairness ***"
    for t in range_ISK:
        algorithm = "SK"
        res = submodular_optimization(unfairness_data, accuracy_data, t, algorithm)
        print "Maximize accuracy s.t. unfairness <= {}".format(t)
        if res[0] is None:
            print "No feature set satisfies these constraints."
        else:
            print "Feature set: {}, accuracy = {}, unfairness = {}\n".format(",".join(res[0]), round(res[2],2), round(res[3],2))


def main():
    demo()
         
    
if __name__=="__main__":
    main()
