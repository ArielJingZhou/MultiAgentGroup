# -*- coding: utf-8 -*-
import sys
import csv
import time
import pickle
import numpy as np
import warnings

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib


warnings.filterwarnings("ignore") # Some depreciate warnings regarding scikit in online learning

f = open('validation.csv', 'r')
rows_validation = f.readlines()[1:]
f.close()
'''
f = open('test.csv', 'r')
rows_validation = f.readlines()[1:]
f.close()
bid_ids = []
for row in rows_validation:
    bid_ids.append(row.split(',')[2])
print(bid_ids[0])
'''
#
# PROCESS DATA
#
def process_data():
    # Build dataframes
    X_dict = []
    y = []

    # Open file
    with open('train.csv', 'r') as csvfile:
        # Create reader
        reader = csv.DictReader(csvfile)
        for row in reader:
            y.append(int(row['click']))            
            X_dict.append({'weekday': row['weekday'], 'hour': row['hour'], 'region': row['region'],\
                           'city': row['city'], \
                           'slotformat': row['slotformat'], 'slotprice': row['slotprice'],\
                           'advertiser': row['advertiser'], 'useragent': row['useragent'], 'adexchange': row['adexchange'], \
                           'domain': row['domain'], 'url': row['url'], 'slotvisibility': row['slotvisibility'], 'keypage': row['keypage'], \
                           'usertag': row['usertag'], 'creative': row['creative']})
    print(len(X_dict))
    return X_dict, y

def process_test_data():
    X_dict = []

    # Open file
    with open('validation.csv', 'r') as csvfile:
        # Create reader
        reader = csv.DictReader(csvfile)

        for row in reader:            
            X_dict.append({'weekday': row['weekday'], 'hour': row['hour'], 'region': row['region'],\
                           'city': row['city'], \
                           'slotformat': row['slotformat'], 'slotprice': row['slotprice'],\
                           'advertiser': row['advertiser'], 'useragent': row['useragent'], 'adexchange': row['adexchange'], \
                           'domain': row['domain'], 'url': row['url'], 'slotvisibility': row['slotvisibility'], 'keypage': row['keypage'], \
                           'usertag': row['usertag'], 'creative': row['creative']})
    print(len(X_dict))
    return X_dict


def decision_tree(load_model):
    start = time.time()
    # Creating test set and turn into one-hot encoded vectors
    X_dict_test = process_test_data()
    dict_one_hot_encoder = DictVectorizer(sparse=False)
    # Create training set
    X_dict_train, y_train = process_data()
    dict_one_hot_encoder.fit(X_dict_train)
    X_test = dict_one_hot_encoder.transform(X_dict_test)

    # Load Model
    if load_model == True:
        print('✔  Loading model from previous training...')
        d_tree_file = open('decision_tree_model_allfeatures.sav', 'rb')
        decision_tree_final = pickle.load(d_tree_file)
        # d_tree_file.close()

        # Evaluate model on test set
        prob = decision_tree_final.predict_proba(X_test)[:, 1]
        f = open('testing_bidding_price.csv', 'w')
        f.write('bidid,bidprice\n')
        base_bid = 80
        bid_prices = base_bid * prob / 1793 * 2430981
        for i in range(len(bid_ids)):
            bid_id = bid_ids[i]
            bid_price = bid_prices[i]

            f.write(bid_id + ',' + str(bid_price) + '\n')
        f.close()
        print(bid_prices[0])
        d_tree_file.close()
        return 0

    if load_model == False:
        print("*  Decision tree model training started...")


    # Transform training dictionary into one-hot encoded vectors
    X_train = dict_one_hot_encoder.transform(X_dict_train)
    print(len(X_train[0]))

    # Train decision tree classifier
    params = {'max_depth': [3, 10, None]}
    decision_tree_model = DecisionTreeClassifier(criterion='gini',
                                                 min_samples_split=30)
    grid_search = GridSearchCV(decision_tree_model, params, n_jobs=-1, cv=3, scoring='roc_auc')
    print("Training started..")
    grid_search.fit(X_train, y_train)
    print('✔  Decision tree model training complete..."\t\t{0:.1f}s'.format(time.time() - start))

    # Use model with best parameter as final model
    decision_tree_final = grid_search.best_estimator_

    # Save Model
    decision_tree_model_file = open('decision_tree_model_allfeatures.sav', "wb")
    pickle.dump(decision_tree_final, decision_tree_model_file)
    decision_tree_model_file.close()
    print('✔  Decision tree model saved...')

    # Evaluate and run model on training data
    prob = decision_tree_final.predict_proba(X_test)[:, 1]
    f = open('decision_tree_tune_base_bid_allfeatures.csv', 'w')
    f.write('base_bid,clicks\n')
    for base_bid in range(50, 90, 1):
        score = bidding(prob, base_bid)
        print(str(base_bid) + ',' + str(score))
        f.write(str(base_bid) + ',' + str(score) + '\n')
    f.close()
    print('✔  clicks on test set:' + str(score))


def bidding(pCTRs, base_bid):
    bid_prices = base_bid * pCTRs / 1793 * 2430981
    i = 0
    clicks = 0
    spend = 0
    for row in rows_validation:
        pay_price = int(row.split(',')[21])
        bid_price = bid_prices[i]
        if bid_price >= pay_price:
            spend += pay_price
            if row.split(',')[0] == '1':
                clicks += 1
            if spend > 6250000:
                break
        i += 1
    return clicks


#
# MAIN
#
def main():
    # Initial Message
    print("Click-through rate models training started...\n")
    
    # Decision Tree
    printGreen('Decision Tree')
    decision_tree(load_model=False)
    print('\n')
    
    print("✔  Done")

if __name__ == '__main__':
    main()
