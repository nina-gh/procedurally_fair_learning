# Fairness Beyond Non-discrimination: Feature Selection for Fair Decision Making

This repository provides a python implementation of our mechanism for procedurally fair feature selection, as well as the accompanying dataset.


### Demo

By running demo.py, you can see a demo of our mechanism for Procedurally Fair Feature Selection, applied on a recidivism risk prediction task, on the dataset considered by ProPublica relating to the COMPAS system, from [3].

The demo showcases the results of minimizing unfairness s.t. a constraint on accuracy, as well as maximizing accuracy s.t. a constraint on unfairness, for a range of thresholds on accuracy and unfairness.

For the task of minimizing unfairness s.t. a constraint on accuracy, our mechanism tries to identify the set of features that minimizes the unfairness of the classifier, while achieving an accuracy greater or equal than the given threshold. 

For the task of maximizing accuracy s.t. a constraint on unfairness, our mechanism tries to identify the set of features that maximizes the accuracy of the classifier, while achieving an unfairness smaller or equal to the given threshold.

Please note that, as explained in [2], some of the results are infeasible, i.e. they don't satisfy the constraint.

For each of the tasks, the demo prints the selected set of features, its accuracy and unfairness. 


### References

[1] Julia Angwin, Jeff Larson, Surya Mattu, and Lauren Kirchner.
    Machine Bias: There’s Software Used Across the Country to Predict Future Criminals. And it’s Biased Against Blacks.
    https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing (2016).

[2] Nina Grgi&#263;-Hla&#269a, Muhammad Bilal Zafar, Krishna P. Gummadi, and Adrian Weller. 
	Beyond Distributive Fairness in Algorithmic Decision Making: Feature Selection for Procedurally Fair Learning
	AAAI (2018).

[3] Rishabh Iyer and Jeff A. Bilmes. 
    Submodular Optimization with Submodular Cover and Submodular Knapsack Constraints. 
    NIPS (2013).


### Files

1. code/demo.py
2. code/submodular_optimization.py
3. code/process_fairness.py
4. code/classification.py		
5. code/measure_increase_decrease_disp_acc.py	
6. code/preprocess_classification_datasets.py	

7. data/ProPublica_amt_data.csv	
8. data/compas-scores-two-years.csv				
9. data/ProPublica_increase_decrease_disp_acc_null_set.pickle
10. data/ProPublica_clf_data.pickle			


### File Descriptions

#### Code

- File 1: File 1 contains a demo of our mechanism for Feature Selection for Fair Decision Making.

- File 2: File 2 contains an implementation of the ISSC and ISK algorithms for Submodular Optimization with Submodular Cover and Submodular Knapsack Constraints, from [3].

- Files 3 and 5: File 3 contains an implementation of the feature-apriori, feature-accuracy and feature-disparity measures of fairness, from [2]. File 5 contains utility functions for calculating these measures, which calculate marginal gains in accuracy and disparate mistreatment.

- Files 4 and 6: File 4 calculates the classification metrics of a dataset, based on the sklearn implementation of logistic regression. File 6 contains utility functions for preprocessing the classification data.


#### Data

- File 7: File 7 contains the results of an Amazon Mechanical Turk (AMT) annotation tasks. The task was to judge if a feature is fair to use, in three different scenarios: (i) a priori, (ii) if it increases the accuracy of classification, (iii) if it increases a measure of disparity of the classifier. The survey instrument can be found here: http://twitter-app.mpi-sws.org/process_fairness_web_surveys/ProPublica/survey.php

- File 8: File 8 contains the dataset considered by ProPublica relating to the COMPAS system, from [1], originally found here: https://github.com/propublica/compas-analysis

- File 9: File 9 contains the output of File 5, for File 8 as the input.

- File 10: File 10 contains the output of File 6, for File 8 as the input.


### File Formats

- File 7: File 7 is a csv file that contains the responses of AMT workers to the survey questions. A response of "1" means that the AMT worker marked the feature as fair to use, in that scenario, while "-1" means that it was marked as unfair to use.

- File 8: File 8 is a csv file containing information about all criminal defendants who were subject to screening by COMPAS, a commercial recidivism risk assessment tool, in Broward County, Florida, during 2013 and 2014.

- File 9: File 9 is a pickle file containing a dictionary, which contains two dictionaries, one for accuracy and one for disparity. The keys of these dictionaries are the features and the values are 0 (False) or 1 (True), depending whether the given feature increases accuracy/disparity for more than a given threshold epsilon.

- File 10: File 9 is a pickle file containing a dictionary. The keys are 9 of the features from file 8, the class label,  "classification_features" and "continuous_features". For the 9 features and the class label, the values are numpy arrays containing corresponding feature vectors. "classification_features" and "continuous_features" contain lists of features, where "classification_features" contains a list of the 9 features we are using in the classification task, and "continuous_features" contains a subset of them, which are continuous.
