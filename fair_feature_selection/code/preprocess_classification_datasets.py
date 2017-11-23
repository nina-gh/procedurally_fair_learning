import pandas as pd
import numpy as np
import pickle

DATA_FOLDER = "../data/"


def preprocess_ProPublica_classification_data():
    '''
    :return: preprocessed ProPublica COMPAS dataset
    '''
    CLASS_LABEL = "two_year_recid"
    CLASSIFICATION_FEATURES = ["age", "sex", "race", "c_charge_degree", "c_charge_desc", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]
    CONTINUOUS_FEATURES = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]
    
    data = dict()
    data["classification_features"] = CLASSIFICATION_FEATURES
    data["continuous_features"] = CONTINUOUS_FEATURES 
    
    df = pd.read_csv(DATA_FOLDER + "compas-scores-two-years.csv")
    data_tmp = df.to_dict('list')
    for k in data_tmp.keys():
        if k == CLASS_LABEL:
            data["class_label"] = np.array(data_tmp[k])
        elif k in CLASSIFICATION_FEATURES:        
            data[k] = np.array(data_tmp[k])
                       
    # for feature "race": reduce to Caucasian and non Caucasian
    data['race'][data['race']!='Caucasian'] = 0 # non Caucasian 
    data['race'][data['race']=='Caucasian'] = 1 # Caucasian

    # for feature "c_charge_desc": reduce to 10 most freq and other
    CHARGE_DESCRIPTIONS = {'dui': ['driving under the influence', 'dui property damage/injury'],
                       'license': ['driving while license revoked', 'felony driving while lic suspd', 'driving license suspended'],
                       'felony petit theft': ['felony petit theft'],
                       'battery': ['battery', 'aggravated assault w/dead weap', 'felony battery (dom strang)', 'aggrav battery w/deadly weapon'],
                       'pos': ['possession of cocaine', 'pos cannabis w/intent sel/del', 'possess cannabis/20 grams or less', 'possession of cannabis', 'poss3,4 methylenedioxymethcath'],
                       'burglary': ['burglary unoccupied dwelling', 'burglary conveyance unoccup'], 'grand theft (motor vehicle)': ['grand theft (motor vehicle)'],
                       'arrest case no charge': ['arrest case no charge'], 'grand theft in the 3rd degree': ['grand theft in the 3rd degree']}    
    
    for i, element in enumerate(data["c_charge_desc"]):
        element = element.lower()

        for key in CHARGE_DESCRIPTIONS:
            if element in CHARGE_DESCRIPTIONS[key]:
                data["c_charge_desc"][i] = key
        
        if data["c_charge_desc"][i] not in CHARGE_DESCRIPTIONS.keys():
            data["c_charge_desc"][i] = "other"

    return data
    

def main():
    print "preprocess_classification_data"
    data = preprocess_ProPublica_classification_data()
    pickle.dump(data, open(DATA_FOLDER+"ProPublica_clf_data.pickle", "w"))
    

if __name__=="__main__":
    main() 