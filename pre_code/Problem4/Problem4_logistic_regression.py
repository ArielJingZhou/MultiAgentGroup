# -*- coding: utf-8 -*-
import csv
import time
import pickle
import numpy as np
import warnings

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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
    # Create Logistic Regression Classifier
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