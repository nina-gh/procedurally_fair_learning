from __future__ import division
from process_fairness import calculate_feature_apriori_fairness, calculate_feature_accuracy_fairness, calculate_feature_disparity_fairness
from classification import calculate_classification_metrics
import sys

null_accuracy = 0

#######################################
### unfairness & accuracy functions ###
#######################################

def calculate_process_unfairness(unfairness_data, feature_set):
    '''
    :param unfairness_data: data for calculating the values of the submodular unfairness function
    :type unfairness_data: dict
    :param feature_set: argument of the submodular unfairness function
    :type feature_set: list
    :return: process unfairness \in \left[ 0, 1 \right], where 0 is completely fair and 1 is completely unfair
    '''
    
    fairness_definition = unfairness_data["fairness_definition"]
    
    if fairness_definition == "apriori":
        return 1 - calculate_feature_apriori_fairness(unfairness_data, feature_set)
    elif fairness_definition == "accuracy":
        increase_decrease_data = unfairness_data["increase_decrease_accuracy"]
        return 1 - calculate_feature_accuracy_fairness(unfairness_data, feature_set, increase_decrease_data)
    elif fairness_definition == "disparity":
        increase_decrease_data = unfairness_data["increase_decrease_disparity"]
        return 1 - calculate_feature_disparity_fairness(unfairness_data, feature_set, increase_decrease_data)

    else:
        return -1
    

def calculate_accuracy(accuracy_data, feature_set):
    '''
    :param accuracy_data: data for calculating the values of the submodular accuracy function
    :type accuracy_data: dict
    :param feature_set: argument of the submodular unfairness function
    :type feature_set: list
    :return: normalized classification accuracy \in \left[ 0, 1 - null_accuracy \right]
    '''
    
    # normalizing accuracy
    return calculate_classification_metrics(accuracy_data, feature_set)["accuracy"] - null_accuracy


#########################################
### modular upper bound of unfairness ###
#########################################
def modular_upper_bound_of_unfairness(Y, X, unfairness_data):
    '''
    :param Y: argument of the modular unfairness function
    :type Y: list
    :param X: modular upper bound of unfairness is tight at X
    :type X: list
    :param unfairness_data: data for calculating the values of the submodular unfairness function
    :type: dict
    :return: value of the modular function for the argument Y
    '''

    sum_on_X = 0
    for j in X:
        if j not in Y:
            X_wo_j = [x for x in X if x!=j]
            delta = calculate_process_unfairness(unfairness_data, X) - calculate_process_unfairness(unfairness_data, X_wo_j)
            sum_on_X += delta

    sum_on_Y = 0
    for j in Y:
        if j not in X:
            delta = calculate_process_unfairness(unfairness_data, [j]) - calculate_process_unfairness(unfairness_data, [])
            sum_on_Y += delta
            
    result = calculate_process_unfairness(unfairness_data, X) - sum_on_X + sum_on_Y

    return result


#######################################
### modular lower bound of accuracy ###
#######################################
def modular_lower_bound_of_accuracy(Y, unfairness_data, accuracy_data, t, algorithm):
    '''
    :param Y: argument of the modular accuracy function
    :type Y: list
    :param unfairness_data: values of the modular unfairness function, i.e. costs
    :type unfairness_data: dict
    :param accuracy_data: data for calculating the values of the submodular accuracy function
    :type accuracy_data: dict
    :param t: threshold on accuracy or unfairness
    :type t: float
    :param algorithm: SSC or SK
    :type algorithm: string
    :return: greedy permutation for calculating the modular lower bound of accuracy, for the SSC algorithm
    '''
    
    if algorithm == "SSC":
        permutation = greedy_permutation_SSC(unfairness_data, accuracy_data, t)
    elif algorithm == "SK":
        permutation = greedy_permutation_SK(unfairness_data, accuracy_data, t)
    else:
        return -1

    accuracy_values = dict()
    for i in range(len(permutation)+1):
        accuracy_values[i] = calculate_accuracy(accuracy_data, permutation[:i])

    result = dict()
    for feature in Y:
        i = permutation.index(feature)
        delta = accuracy_values[i+1] - accuracy_values[i]
        result[feature] = delta

    return result


def greedy_permutation_SSC(unfairness_data, accuracy_data, t):
    '''
    :param unfairness_data: values of the modular unfairness function, i.e. costs of individual features
    :type unfairness_data: dict
    :param accuracy_data: data for calculating the values of the submodular accuracy function
    :type accuracy_data: dict
    :param t: threshold on accuracy
    :type t: float
    :return: greedy permutation for calculating the modular lower bound of accuracy, for the SSC algorithm
    '''

    permutation = list()
    F = list(accuracy_data["classification_features"])

    while True:
        ob_func_values = dict()
        acc_wo_j = calculate_accuracy(accuracy_data, permutation)

        for j in F:
            acc_w_j = calculate_accuracy(accuracy_data, permutation+[j])
            if acc_w_j < t:
                delta = acc_w_j - acc_wo_j
                # fixing division by zero
                if delta == 0:
                    ob_func_values[j] = sys.float_info.max
                else:
                    ob_func_values[j] = unfairness_data[j]/delta

        if not len(ob_func_values):
            break
                           
        minimizing_j = min(ob_func_values, key=ob_func_values.get)
        permutation.append(minimizing_j)
        F.remove(minimizing_j)

                           
    permutation.extend([f for f in F])

    return permutation


def greedy_permutation_SK(unfairness_data, accuracy_data, t):
    '''
    :param unfairness_data: values of the modular unfairness function, i.e. costs of individual features
    :type unfairness_data: dict
    :param accuracy_data: data for calculating the values of the submodular accuracy function
    :type accuracy_data: dict
    :param t: threshold on accuracy
    :type t: float
    :return: greedy permutation for calculating the modular lower bound of accuracy, for the SSC algorithm
    '''

    permutation = list()
    F = list(accuracy_data["classification_features"])

    while True:
        ob_func_values = dict()
        unf_wo_j = sum([unfairness_data[j] for j in permutation])
        acc_wo_j = calculate_accuracy(accuracy_data, permutation)

        for j in F:
            unf_w_j = unf_wo_j + unfairness_data[j]
            acc_w_j = calculate_accuracy(accuracy_data, permutation+[j])
            if unf_w_j <= t:
                delta = acc_w_j - acc_wo_j
                # fixing division by zero
                if unfairness_data[j] == 0:
                    ob_func_values[j] = sys.float_info.max
                else:
                    ob_func_values[j] = delta/unfairness_data[j]

        if not len(ob_func_values):
            break
                           
        maximizing_j = max(ob_func_values, key=ob_func_values.get)
        permutation.append(maximizing_j)
        F.remove(maximizing_j)

                           
    permutation.extend([f for f in F])

    return permutation


###################
### ISSC and ISK ##
###################
def calculate_cost_and_value(X, unfairness_data, accuracy_data, t, algorithm):
    '''
    :param X: modular upper bound of unfairness is tight at X
    :type X: list
    :param unfairness_data: data for calculating the values of the submodular unfairness function
    :type unfairness_data: dict
    :param accuracy_data: data for calculating the values of the submodular accuracy function
    :type accuracy_data: dict
    :param t: threshold on accuracy or unfairness
    :type t: float
    :param algorithm: SSC or SK
    :type algorithm: string
    :return: data for knapsack or minimization knapsack algorithm
    '''
    
    F = list(accuracy_data["classification_features"])
    knapsack_data = list()
    
    
    modular_unfairness = dict()
    for feature in F:
        modular_unfairness[feature] = modular_upper_bound_of_unfairness([feature], X, unfairness_data) - modular_upper_bound_of_unfairness([], X, unfairness_data)
    modular_accuracy = modular_lower_bound_of_accuracy(F, modular_unfairness, accuracy_data, t, algorithm)

    for feature in F:
        cost = max(modular_unfairness[feature], 0)
        value = max(modular_accuracy[feature], 0)
        knapsack_data.append(tuple([cost, value, feature]))
    
    return knapsack_data

    
def submodular_optimization(unfairness_data, accuracy_data, t, algorithm):
    '''
    :param unfairness_data: data for calculating the values of the submodular unfairness function
    :type unfairness_data: dict
    :param accuracy_data: data for calculating the values of the submodular accuracy function
    :type accuracy_data: dict
    :param t: threshold on accuracy or unfairness
    :type t: float
    :param algorithm: SSC or SK
    :type algorithm: string
    :return: features selected by the submodular optimization algorithm
    '''

    # calculating classification accuracy of null classifier, for normalizing the submodular accuracy function
    global null_accuracy
    null_accuracy = calculate_classification_metrics(accuracy_data, [])["accuracy"]

    # since we are normalizing accuracy, we need to subtract it from t as well
    if algorithm=="SSC":
        t -= null_accuracy

    # start of submodular optimization algorithm
    starting_X = None
    resulting_X = list()
    achieved_unf_vals = dict()
    achieved_acc_vals = dict()
    
    for iteration in range(25):
        starting_X = resulting_X
        
        knapsack_data = calculate_cost_and_value(starting_X, unfairness_data, accuracy_data, t, algorithm)
        # we need ints for knapsack, 3 decimal point precision
        knapsack_data = [tuple([int(round(cost,3)*1000), int(round(value,3)*1000), feature]) for cost, value, feature in knapsack_data]

        if algorithm=="SSC":
            resulting_X = minimization_knapsack(knapsack_data, int(t*1000))
        elif algorithm=="SK":
            resulting_X = knapsack(knapsack_data, int(t*1000))


        if (resulting_X is None) or (tuple(sorted(resulting_X)) in achieved_acc_vals.keys()):
            break
        
        achieved_unf_vals[tuple(sorted(resulting_X))] = calculate_process_unfairness(unfairness_data, resulting_X)
        achieved_acc_vals[tuple(sorted(resulting_X))] = calculate_accuracy(accuracy_data, resulting_X)
            
    resulting_X = find_optimal(achieved_unf_vals, achieved_acc_vals, t, algorithm)[0]
    unf_resulting_X = achieved_unf_vals.get(resulting_X, -1)
    acc_resulting_X = achieved_acc_vals.get(resulting_X, -1) + null_accuracy
    return resulting_X, iteration+1, acc_resulting_X, unf_resulting_X


#############################
### find optimal solution ###
#############################
def find_optimal(unfairness_data, accuracy_data, t, algorithm):
    '''
    :param unfairness_data: values of unfairness of some feature sets
    :type unfairness_data: dict
    :param accuracy_data: values of accuracy of some feature sets
    :type accuracy_data: dict
    :param t: threshold on accuracy or unfairness
    :type t: float
    :param algorithm: SSC or SK
    :type algorithm: string
    :return: optimal feature set
    '''
    
    if algorithm == "SSC":
        sorted_objective_function_values = sorted(unfairness_data.items(), key=lambda x:x[-1])
    elif algorithm == "SK":
        sorted_objective_function_values = sorted(accuracy_data.items(), key=lambda x:-x[-1])
    else:
        return -1

    optimal_feature_set = sorted_objective_function_values[0][0]

    '''
    # removes infeasible results
    optimal_feature_set = None

    for feature_set, val in sorted_objective_function_values:
        if accuracy_data[feature_set] >= t and algorithm == "SSC":
            optimal_feature_set = feature_set
            break
        
        elif unfairness_data[feature_set] <= t and algorithm == "SK":
            optimal_feature_set = feature_set
            break
    '''

    if optimal_feature_set is None:
        return None, None, None

    return optimal_feature_set, unfairness_data[optimal_feature_set], accuracy_data[optimal_feature_set]


#########################################
### dynamic programming: 1/0 knapsack ###
#########################$###############
def minimization_knapsack(minimization_items, minimization_t):
    '''
    :param minimization_items: data for 1/0 minimization knapsack
    :type minimization_items: list
    :param minimization_t: threshold on unfairness or accuracy
    :type minimization_t: int
    :return: optimal feature set
    '''
    
    items = [tuple([accuracy, unfairness, feature]) for unfairness, accuracy, feature in minimization_items]
    t = max(sum([item[0] for item in items]) - minimization_t, 0)
    knapsack_results = knapsack(items, t)
    if knapsack_results is None:
        return None
    
    optimal_feature_set = [item[-1] for item in items if item[-1] not in knapsack_results]

    return optimal_feature_set


def knapsack(items, t):
    '''
    :param items: data for 1/0 knapsack
    :type items: list
    :param t: threshold on unfairness or accuracy
    :type t: int
    :return: optimal feature set
    '''
    
    table = dict()

    for table_index in range(len(items) + 1):
        table[table_index] = dict()
        item_index = table_index-1
        
        for threshold in range(t + 1):
            if table_index == 0:
                table[table_index][threshold] = 0
            
            elif items[item_index][0] > threshold:
                table[table_index][threshold] = table[table_index-1][threshold]
            else:
                table[table_index][threshold] = max(table[table_index-1][threshold], table[table_index-1][threshold - items[item_index][0]] + items[item_index][1])

    # backtracking
    optimal_feature_set = list()
    threshold = t

    for table_index in range(len(items), 0, -1):
        item_index = table_index-1
        if table[table_index][threshold] != table[table_index-1][threshold]:
            optimal_feature_set.append(items[item_index][-1])
            threshold -= items[item_index][0]

    return optimal_feature_set

        
def main():
    print "submodular_optimization"


if __name__=="__main__":
    main()
