import numpy as np
import pandas as pd
import pickle

DATA_FOLDER = "../data/"
PLOT_FOLDER = "../plots/"


#######################
## utility functions ##
#######################

def save_preprocessed_classification_data(data, dataset_name):
    pickle.dump(data, open("{}{}_clf_data.pickle".format(DATA_FOLDER, dataset_name), "w"))


def load_preprocessed_classification_data(dataset_name, show_preview = False):
    data = pickle.load(open("{}{}_clf_data.pickle".format(DATA_FOLDER, dataset_name), "r"))

    if show_preview:
        print "* Preview of preprocessed {} data:".format(dataset_name)
        for key in data:
            print "\t* {}: {}".format(key, np.array_repr(np.array(sorted(set(data[key]))), max_line_width=75))
        print "\n"

    return data


########################
## preprocessing data ##
########################

def preprocess_classification_data(dataset_name, include_neutral=True):
    '''
    :return: preprocessed dataset
    '''
    inherent_properties = ["volitionality", "reliability", "privacy", "relevance", "causes_outcome", "caused_by_sensitive_feature", "causal_loop", "causes_disparity_in_outcomes"]
    features = ["the_defendant's_current_criminal_charge", "the_defendant's_criminal_history", "the_criminal_history_of_the_defendant's_family_and_friends", "the_defendant's_history_of_substance_abuse", "the_stability_of_the_defendant's_employment_and_living_situation", 'the_safety_of_the_neighborhood_the_defendant_lives_in', "the_defendant's_education_and_behavior_in_school", "the_quality_of_the_defendant's_social_life_and_free_time", "the_defendant's_personality", "the_defendant's_beliefs_about_criminality"]
    fairness = "fairness"
    attention_check = "attention_check"

    CLASS_LABEL = fairness
    CONTINUOUS_FEATURES = list(inherent_properties) 
    CLASSIFICATION_FEATURES = list(CONTINUOUS_FEATURES + ["feature", "worker"])
    
    data = dict()
    data["classification_features"] = CLASSIFICATION_FEATURES 
    data["continuous_features"] = CONTINUOUS_FEATURES 
    
    df = pd.read_csv("{}pt_{}_data.csv".format(DATA_FOLDER, dataset_name))
    data_tmp = df.to_dict('list')

    # worker ids
    data["worker"] = list()
    for i, worker in enumerate(data_tmp["worker_id"]):
        for feature in features:
            data["worker"].append(worker)

    data["worker"] = np.array(data["worker"])

    # class label, i.e. fairness judgment. binarized + real thing
    data["class_label"] = list()
    data["fairness"] = list()
    data["feature"] = list()

    for i, worker in enumerate(data_tmp["worker_id"]):
        for feature in features:
            judgment = data_tmp["fairness__{}".format(feature)][i]
            data["fairness"].append(judgment)
            data["feature"].append(feature)

            if judgment < 4:
                data["class_label"].append(-1)
            elif judgment >= 4:
                data["class_label"].append(1)
            else: 
                data["class_label"].append(0)

    data["class_label"] = np.array(data["class_label"]) 
    data["fairness"] = np.array(data["fairness"])
    data["feature"] = np.array(data["feature"])


    for inherent_property in inherent_properties:
        data[inherent_property] = list()
        for i, worker in enumerate(data_tmp["worker_id"]):
            for feature in features:
                data[inherent_property].append(data_tmp["{}__{}".format(inherent_property, feature)][i])

        data[inherent_property] = np.array(data[inherent_property])


    # removing workers that failed attention check
    passed_attention_check = np.array(data_tmp["worker_id"])[np.array(data_tmp["attention_check"])]
    rows_to_filter = np.in1d(data["worker"], passed_attention_check)
    for k in data:
        if k not in ["classification_features", "continuous_features"]:
            data[k] = data[k][rows_to_filter]
    

    # removing all with neutral rating
    if not include_neutral:
        satisfy_condition = np.array(data["fairness"]!=4)

        for k in data:
            if k not in ["classification_features", "continuous_features"]:
                data[k] = data[k][satisfy_condition]

    return data


##################
## all datasets ##
##################

def preprocess_all_datasets():
    dataset_names = ["AMT", "SSI"]

    for include_neutral in [True, False]:
        if include_neutral == False:
            suffix = "_wo_neutral"
        else:
            suffix = ""
            
        for dataset_name in dataset_names:
            data = preprocess_classification_data(dataset_name, include_neutral=include_neutral)
            save_preprocessed_classification_data(data, dataset_name+suffix)


##########
## main ##
##########

def main():
    print "preprocess_classification_datasets.py"


if __name__=="__main__":
    main() 