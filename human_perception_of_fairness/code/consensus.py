from __future__ import division
import numpy as np
import scipy
import math
from krippendorff_alpha import krippendorff_alpha
from diptest import diptest
from collections import Counter, defaultdict


features = {
"the_defendant's_current_criminal_charge": "Current Charges", 
"the_defendant's_criminal_history": "Criminal History: self",
"the_criminal_history_of_the_defendant's_family_and_friends": "Criminal History: associates", 
"the_defendant's_history_of_substance_abuse": "Substance Abuse", 
"the_stability_of_the_defendant's_employment_and_living_situation": "Stability of Employment", 
'the_safety_of_the_neighborhood_the_defendant_lives_in': "Neighborhood Safety", 
"the_defendant's_education_and_behavior_in_school": "Education \& School Behavior", 
"the_quality_of_the_defendant's_social_life_and_free_time": "Quality of Social Life", 
"the_defendant's_personality": "Personality",
"the_defendant's_beliefs_about_criminality": "Criminal Attitude"
}

inherent_properties = {
"volitionality": "Volitionality", 
"reliability": "Reliability", 
"privacy": "Privacy", 
"relevance": "Relevance", 
"causes_outcome": "Causes outcome", 
"caused_by_sensitive_feature": "Caused by sensitive feature", 
"causal_loop": "Causes vicious cycle", 
"causes_disparity_in_outcomes": "Causes disparity in outcomes",
"predicted_fairness": "Predicted fairness", 
"fairness": "Fairness" 
}


def calculate_concensus(data):
    # calculating consensus
    results = dict()
    for inherent_property in inherent_properties:
        results[inherent_property] = dict()
        for feature in features: 
            results[inherent_property][feature] = dict()

            answers = data[inherent_property][data["feature"]==feature]

            # distribution, 7 valued
            answer_counts = Counter(answers)
            distribution = list()

            if inherent_property == "predicted_fairness":
                value_range = [-1,1]
            else:
                value_range = range(1,8)

            for value in value_range:
                distribution.append(answer_counts.get(value, 0)/answers.shape[0])
            distribution = np.array(distribution)
            results[inherent_property][feature]["distribution"] = np.array(distribution)

            # distribution, 3 valued
            if inherent_property != "predicted_fairness":
                ternary_distribution = np.array([np.sum(distribution[:3]), distribution[3], np.sum(distribution[4:])])
                results[inherent_property][feature]["ternary_distribution"] = np.array(ternary_distribution)
            else:
                ternary_distribution = np.array(distribution)
                results[inherent_property][feature]["ternary_distribution"] = np.array(ternary_distribution)

            # calcualations
            results[inherent_property][feature]["mean"] = np.mean(answers)
            results[inherent_property][feature]["variance"] = np.var(answers)

            results[inherent_property][feature]["entropy"] = scipy.stats.entropy(distribution)
            results[inherent_property][feature]["normalized_entropy"] = 1 - (scipy.stats.entropy(distribution) / np.log(len(distribution)))
            results[inherent_property][feature]["ternary_entropy"] = scipy.stats.entropy(ternary_distribution)
            results[inherent_property][feature]["ternary_normalized_entropy"] = 1 - (scipy.stats.entropy(ternary_distribution) / np.log(len(ternary_distribution)))
                        
    print_tables_with_results(results)
    return results


def print_tables_with_results(results):
    # printing consensus on fairness
    print "\n\n\n"
    print "\nprinting consensus on fairness\n"
    inherent_property = "fairness"
    print "&  & \\multicolumn{9}{c|}{\\textbf{Probability distribution over answers}} & \\multicolumn{2}{c}{\\textbf{Consensus}} \\\\"
    print "\\textbf{Feature} & \\textbf{Mean fairness} & \\textbf{1} & \\textbf{2} & \\textbf{3} & \\textbf{1-3} & \\textbf{4} & \\textbf{5-7} & \\textbf{5} & \\textbf{6} & \\textbf{7} & \\textbf{7 point} & \\textbf{3 point} \\\\"
    print "\\hline"
    
    mean_fairness_vals = {feat:results[inherent_property][feat]["mean"] for feat in features}
    sorted_pairs = sorted(mean_fairness_vals.items(), key=lambda x:-x[-1])
    sorted_features = [i for i,j in sorted_pairs]

    for feature in sorted_features:
        res = results[inherent_property][feature]
        print " & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}\\\\".format(\
        features[feature], round(res["mean"], 2), \
        round(res["distribution"][0], 2), round(res["distribution"][1], 2), \
        round(res["distribution"][2], 2), round(res["ternary_distribution"][0], 2), \
        round(res["distribution"][3], 2), round(res["ternary_distribution"][-1], 2), \
        round(res["distribution"][4], 2), round(res["distribution"][5], 2), \
        round(res["distribution"][6], 2),\
        round(res["normalized_entropy"], 2), round(res["ternary_normalized_entropy"], 2))
            

    # printing consensus on fairness, prediced fairness and latent properties
    print "\n\n\n"
    print "\nprinting consensus on fairness, prediced fairness and latent properties\n"
    sorted_inherent_properties = ["fairness", "predicted_fairness", "reliability", "privacy", "relevance", "causes_outcome", "volitionality", "caused_by_sensitive_feature", "causal_loop", "causes_disparity_in_outcomes"] 
    
    print "Feature, {}".format(", ".join([inherent_properties[p] for p in sorted_inherent_properties]))

    for i, feature in enumerate(sorted_features):
        print "{}, {}".format(i, ", ".join([str(round(results[p][feature]["ternary_normalized_entropy"], 3)) for p in sorted_inherent_properties]))

    return 0


if __name__=="__main__":
    print "concensus.py"
