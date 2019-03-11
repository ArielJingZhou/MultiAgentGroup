# -*- coding: utf-8 -*-
import sys
import csv
import time
import pickle
import numpy as np
import warnings

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

from output import printGreen, printYellow, printRed

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

            # Append Label to y
            y.append(int(row['click']))
            # Remove features
            '''
            del row['click'], row['bidid'], row['userid'], row['useragent'], row['IP'], row['adexchange'], \
                row['domain'], row['url'], row['urlid'], row['slotid'], row['slotvisibility'], row['creative'], \
                row['bidprice'], row['payprice'], row['keypage'], row['usertag']
            
            # Append input to X
            
            X_dict.append({'weekday': row['weekday'], 'hour': row['hour'], 'region': row['region'],\
                           'city': row['city'], 'slotwidth': row['slotwidth'], 'slotheight': row['slotheight'],\
                           'slotformat': row['slotformat'], 'slotprice': row['slotprice'],\
                           'advertiser': row['advertiser']})
            '''
            X_dict.append({'slotprice': row['slotprice'], 'useragent': row['useragent'],\
                           'advertiser': row['advertiser']})
    print(X_dict[0])
    return X_dict, y

def process_test_data():
    X_dict = []
    y = []

    # Open file
    with open('validation.csv', 'r') as csvfile:
        # Create reader
        reader = csv.DictReader(csvfile)

        for row in reader:
            y.append(int(row['click']))
            # Remove features
            '''
            del row['click'], row['bidid'], row['userid'], row['IP'], row['adexchange'], \
                row['domain'], row['url'], row['urlid'], row['slotid'], row['slotvisibility'], row['creative'], \
                row['bidprice'], row['payprice'], row['keypage'], row['usertag']

            # Append input to X
            
            X_dict.append({'weekday': row['weekday'], 'hour': row['hour'], 'region': row['region'],\
                           'city': row['city'], 'slotwidth': row['slotwidth'], 'slotheight': row['slotheight'],\
                           'slotformat': row['slotformat'], 'slotprice': row['slotprice'],\
                           'advertiser': row['advertiser']})
            '''
            X_dict.append({'slotprice': row['slotprice'], 'useragent': row['useragent'],\
                           'advertiser': row['advertiser']})

    return X_dict, y


#
# DECISION TREE ~20 min to train
#
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
        printGreen('✔  Loading model from previous training...')
        d_tree_file = open('../models/decision_tree_model_3feature.sav', 'rb')
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
        printYellow("*  Decision tree model training started...")


    # Transform training dictionary into one-hot encoded vectors
    X_train = dict_one_hot_encoder.transform(X_dict_train)
    print(len(X_train[0]))


    # print(len(X_test[0]))



    # Train decision tree classifier
    params = {'max_depth': [3, 10, None]}
    decision_tree_model = DecisionTreeClassifier(criterion='gini',
                                                 min_samples_split=30)
    grid_search = GridSearchCV(decision_tree_model, params, n_jobs=-1, cv=3, scoring='roc_auc')
    print("Training started..")
    grid_search.fit(X_train, y_train)
    printGreen('✔  Decision tree model training complete..."\t\t{0:.1f}s'.format(time.time() - start))

    # Use model with best parameter as final model
    decision_tree_final = grid_search.best_estimator_

    # Save Model
    decision_tree_model_file = open('../models/decision_tree_model2.sav', "wb")
    pickle.dump(decision_tree_final, decision_tree_model_file)
    decision_tree_model_file.close()
    printGreen('✔  Decision tree model saved...')

    # Evaluate and run model on training data
    prob = decision_tree_final.predict_proba(X_test)[:, 1]
    f = open('decision_tree_tune_base_bid.csv', 'w')
    f.write('base_bid,clicks\n')
    for base_bid in range(50, 90, 1):
        score = bidding(prob, base_bid)
        printGreen('✔  clicks on test set:' + str(score))
        f.write(str(base_bid) + ',' + str(score) + '\n')
    f.close()
    printGreen('✔  clicks on test set:' + str(score))

    return 0


def bidding(pCTRs, base_bid):
    bid_prices = base_bid * pCTRs / 1793 * 2430981
    i = 0
    clicks = 0
    spend = 0
    winning_impressions = 0
    for row in rows_validation:
        pay_price = int(row.split(',')[21])
        bid_price = bid_prices[i]
        if bid_price >= pay_price:
            spend += pay_price
            winning_impressions += 1
            if row.split(',')[0] == '1':
                clicks += 1
            if spend > 6250000:
                break
        i += 1
    click_through_rate = "{:.3%}".format(clicks / winning_impressions)

    if clicks == 0:
        average_cpm = 0
        average_cpc = 0
    else:
        average_cpm = spend / winning_impressions * 1000
        average_cpc = spend / clicks

    print(clicks, click_through_rate, spend, average_cpm, average_cpc)

   # print('clicks:' + str(clicks) + "\n")
    return clicks


#
# RANDOM FOREST ~ 20 min to train
#
def random_forest(load_model=False):
    start = time.time()
    if load_model == False:
        printYellow("*  Random forest model training started...")

    # Create training set of 100,000 samples
    X_dict_train, y_train = process_data()

    # Transform training dictionary into one-hot encoded vectors
    dict_one_hot_encoder = DictVectorizer(sparse=False)
    dict_one_hot_encoder.fit(X_dict_train)
    X_train = dict_one_hot_encoder.fit_transform(X_dict_train)
    print("X_train")
    # Creating test set and turn into one-hot encoded vectors
    X_dict_test = process_test_data()
    X_test = dict_one_hot_encoder.transform(X_dict_test)
    print("X_test")
    # Load model instead of training again..
    if load_model == True:
        printGreen('✔  Loading model from previous training...')
        r_forest_file = open('../models/random_forest_model.sav', 'rb')
        random_forest_final = pickle.load(r_forest_file)
        probs = random_forest_final.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, probs)
        printGreen('✔  ROC AUC score on test set: {0:.3f}'.format(score))
        r_forest_file.close()
        return 0

    # Train random forest classifier
    params = {'max_depth': [3, 10, None]}
    random_forest_model = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30,
                                                 n_jobs=-1)
    grid_search = GridSearchCV(random_forest_model, params, n_jobs=-1, cv=3, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    printGreen('✔  Random forest model training complete..."\t\t{0:.1f}s'.format(time.time() - start))

    # Use best paramter for final model
    random_forest_final = grid_search.best_estimator_

    # Save Model
    random_forest_file = open('../models/random_forest_model_3feature.sav', "wb")
    pickle.dump(random_forest_final, random_forest_file)
    random_forest_file.close()
    printGreen('✔  Random forest model saved...')

    # Evaluate model
    probs = random_forest_final.predict_proba(X_test)[:, 1]
    f = open('random_forest_tune_base_bid.csv', 'w')
    f.write('base_bid,clicks\n')
    for base_bid in range(50, 90, 1):
        score = bidding(probs, base_bid)
        printGreen('✔  clicks on test set:' + str(score))
        f.write(str(base_bid) + ',' + str(score) + '\n')
    f.close()
    printGreen('✔  clicks on test set:' + str(score))

    return 0

#
# SGD-BASED LOGISTIC REGRESSION ~20 sec. to train
#
def logistic_regression(sample_size=100000, load_model=True):
    start = time.time()

    if load_model == False:
        printYellow("*  Logistic regression model training started...")

    # Create Training Set
    n = sample_size
    X_dict_train, y_train = process_data()
    dict_one_hot_encoder = DictVectorizer(sparse=False)
    X_train = dict_one_hot_encoder.fit_transform(X_dict_train)

    # Create Test Set
    X_dict_test, y_test = process_test_data()
    X_test = dict_one_hot_encoder.transform(X_dict_test)

    X_train_n = X_train
    y_train_n = np.array(y_train)


    # Load model instead of training again
    if load_model == True:
        printGreen('✔  Loading model from previous training...')
        l_reg_file = open('../models/logistic_regression_model.sav', 'rb')
        log_reg_model = pickle.load(l_reg_file)
        predictions = log_reg_model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, predictions)
        printGreen("✔  ROC AUC score on test set: {0:.3f}".format(score))

        # Evaluate and run model on training data
        f = open('logistic_regression_base_bid.csv', 'w')
        f.write('base_bid,clicks\n')
        for base_bid in range(50, 90, 1):
            score = bidding(predictions, base_bid)
            #printGreen('✔  clicks on test set:' + str(score))
        f.write(str(base_bid) + ',' + str(score) + '\n')
        f.close()
        printGreen('✔  clicks on test set:' + str(score))

        return 0

    '''
    # Create SGD Logistic Regression Classifier
    log_reg_model = LogisticRegression()

    # Train Classifier
    log_reg_model.fit(X_train_n, y_train_n)
    printGreen('✔  Logistic regression model training complete..."\t\t{0:.1f}s'.format(time.time() - start))

    # Run model on test set
    predictions = log_reg_model.predict_proba(X_test)[:, 1]

    # Evaluate model
    score = roc_auc_score(y_test, predictions)
    printGreen("✔  ROC AUC score on test set: {0:.3f}".format(score))

    # Save model
    l_reg_file = open('../models/logistic_regression_model.sav', "wb")
    pickle.dump(log_reg_model, l_reg_file)
    l_reg_file.close()
    printGreen('✔  Logistic regression model saved...')

    '''


#
# LOGISTIC REGRESSION USING ONLINE LEARNING ~6 min. to train
#
def logistic_regression_ol(load_model=True):
    start = time.time()


    if load_model == False:
        printYellow("*  Logistic regression (using online learning) model training started...")

    # Build Classifier
    og_reg_model = LogisticRegression()
    
    # Training sets
    X_dict_train, y_train = process_data()
    dict_one_hot_encoder = DictVectorizer(sparse=False)
    X_train = dict_one_hot_encoder.fit_transform(X_dict_train)
    
    X_train_100k = X_train
    y_train_100k = np.array(y_train)

    # Test sets
    X_dict_test, y_test_next10k = process_test_data()
    X_test_next10k = dict_one_hot_encoder.transform(X_dict_test)

    if load_model == True:
        printGreen('✔  Loading model from previous training...')
        l_reg_file = open('../models/logistic_regression_model_ol.sav', 'rb')
        log_reg_model = pickle.load(l_reg_file)
        predictions = log_reg_model.predict_proba(X_test_next10k)[:, 1]
        score = roc_auc_score(y_test_next10k, predictions)
        printGreen("✔  ROC AUC score on test set: {0:.3f}".format(score))

        # Evaluate and run model on training data
        f = open('logistic_regression_ol_base_bid.csv', 'w')
        f.write('base_bid,clicks\n')
        for base_bid in range(50, 90, 1):
            score = bidding(predictions, base_bid)
            #printGreen('✔  clicks on test set:' + str(score))
        f.write(str(base_bid) + ',' + str(score) + '\n')
        f.close()
        printGreen('✔  clicks on test set:' + str(score))

        return 0

    '''
    # Train and partially fit on 1 million samples
    for i in range(20):
        X_dict_train, y_train_every = process_data()
        X_train_every = dict_one_hot_encoder.transform(X_dict_train)
        og_reg_model.fit(X_train_every, y_train_every)
    
    printGreen('✔  Logistic regression (using online learning) model training complete..."\t\t{0:.1f}s'.format(time.time() - start))
    
    # Get test set
    X_dict_test, y_test_next = process_test_data()
    X_test_next = dict_one_hot_encoder.transform(X_dict_test)
    
    # Evaluate
    predict = og_reg_model.predict_proba(X_test_next)[:, 1]
    score = roc_auc_score(y_test_next, predict)
    printGreen("✔  ROC AUC score on test set: {0:.3f}".format(score))

    # Save Model
    l_reg_file = open('../models/logistic_regression_model_ol.sav', "wb")
    pickle.dump(og_reg_model, l_reg_file)
    l_reg_file.close()
    printGreen('✔  Logistic regression (using online learning) model saved...')
    return 0
    '''
#
# MAIN
#
def main():
    # Initial Message
    printGreen("Click-through rate models training started...\n")
    '''
    # Decision Tree
    printGreen('Decision Tree')
    decision_tree(load_model=True)
    print('\n')
    '''
    '''
    # Random Forest
    printGreen('Random Forest')
    random_forest(load_model=False)
    print('\n')
    '''

    # Logistic Regression
    printGreen('Logistic Regression')
    logistic_regression(load_model=True)
    print('\n')


    # OL Logistic Regression
    printGreen('Logistic Regressions using Online Learning')
    logistic_regression_ol(load_model=True)
    print('\n')


    printGreen("✔  Done")

if __name__ == '__main__':
    main()