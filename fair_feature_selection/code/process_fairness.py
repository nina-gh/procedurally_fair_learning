from __future__ import division


########################
### apriori fairness ###
########################
def calculate_feature_apriori_fairness(unfairness_data, features):
    '''
    :param unfairness_data: data for calculating the values of the submodular unfairness_function
    :type unfairness_data: dict
    :param feature_set: argument of the submodular unfairness function. we are calculating the feature-apriori fairness of this feature set.
    :type feature_set: list
    :return: process fairness \in \left[ 0, 1 \right], where 0 is completely unfair and 1 is completely fair
    '''
    
    all_workers = list(unfairness_data["worker_id"])
    approving_workers = list(all_workers)
    
    for feature in features:
        apriori = list(unfairness_data["{}_apriori".format(feature)])
        workers_apriori = [all_workers[i] for i, x in enumerate(apriori) if x == 1]
        
        approve_feat = list(workers_apriori)
        approving_workers = list(set(approving_workers) & set(approve_feat))
        
    process_fairness = len(approving_workers)/len(all_workers)
    return process_fairness


#########################
### accuracy fairness ###
#########################
def calculate_feature_accuracy_fairness(unfairness_data, features, increase_decrease_data):
    '''
    :param unfairness_data: data for calculating the values of the submodular unfairness_function
    :type unfairness_data: dict
    :param feature_set: argument of the submodular unfairness function. we are calculating the feature-apriori fairness of this feature set.
    :type feature_set: list
    :param increase_decrease_data: data[feature] = 1 if feature increases accuracy for more than \epsilon when added to the empty set, else 0
    :type increase_decrease_data: dict
    :return: process fairness \in \left[ 0, 1 \right], where 0 is completely unfair and 1 is completely fair
    '''
    
    all_workers = list(unfairness_data["worker_id"])
    approving_workers = list(all_workers)
    
    for feature in features:
        apriori = list(unfairness_data["{}_apriori".format(feature)])
        accuracy = list(unfairness_data["{}_accuracy".format(feature)])
        workers_apriori = [all_workers[i] for i, x in enumerate(apriori) if x == 1]
        workers_accuracy = [all_workers[i] for i, x in enumerate(accuracy) if x == 1]
        
        acc_w = float(increase_decrease_data[feature])
        acc_wo = 1 - acc_w
        
        if acc_w > acc_wo: 
            approve_feat = list(set(workers_apriori) | set(workers_accuracy))
        else:
            approve_feat = workers_apriori
        approving_workers = list(set(approving_workers) & set(approve_feat))
        
    process_fairness = len(approving_workers)/len(all_workers)
    return process_fairness
    

##########################
### disparity fairness ###
##########################
def calculate_feature_disparity_fairness(unfairness_data, features, increase_decrease_data):
    '''
    :param unfairness_data: data for calculating the values of the submodular unfairness_function
    :type unfairness_data: dict
    :param feature_set: argument of the submodular unfairness function. we are calculating the feature-apriori fairness of this feature set.
    :type feature_set: list
    :param increase_decrease_data: data[feature] = 1 if feature increases disparity for more than \epsilon when added to the empty set, else 0
    :type increase_decrease_data: dict
    :return: process fairness \in \left[ 0, 1 \right], where 0 is completely unfair and 1 is completely fair
    '''
    
    all_workers = list(unfairness_data["worker_id"])
    approving_workers = list(all_workers)

    for feature in features:
        apriori = list(unfairness_data["{}_apriori".format(feature)])
        print data
        disparity = list(unfairness_data["{}_disparity".format(feature)])
        workers_apriori = [all_workers[i] for i, x in enumerate(apriori) if x == 1]
        workers_disparity = [all_workers[i] for i, x in enumerate(disparity) if x == 1]
        
        disp_w = float(increase_decrease_data[feature])
        disp_wo = 1 - disp_w
                
        if disp_w <= disp_wo:
            approve_feat = list(set(workers_apriori) | set(workers_disparity))
        else:
            approve_feat = workers_disparity

        approving_workers = list(set(approving_workers) & set(approve_feat))
        
    process_fairness = len(approving_workers)/len(all_workers)
    return process_fairness
   

def main():
    print "process_fairness"


if __name__=="__main__":
    main() 
