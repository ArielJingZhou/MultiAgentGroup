import sys
import csv
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
from math import sqrt

# read the data
# file path may need to be changed
train_set = pd.read_csv("data/train.csv")
valid_set = pd.read_csv("data/validation.csv")
#test_set = pd.read_csv("data/test.csv") 

# downsample train set, too big
def downsampling(data):
    no_click = data.query('click == 0')
    do_click = data.query('click == 1')
    nums = len(do_click) * 700
    new_no_click = no_click.sample(n=nums, random_state=42)
    return pd.concat([new_no_click, do_click])

# drop user tag for problem 3 ... to be considered more
# slotprice is considered as continuous variable

def data_preprocessing(data, enforce_cols = None):
    #data = data.sort_index(axis=0)
    
    # drop features
    to_drop_columns = ['bidid', 'keypage', 'userid', 'url', 'urlid',
                       'IP', 'domain', 'slotid', 'creative', 'usertag']
    data = data.drop(to_drop_columns, axis=1)
    
    # one hot encoding categorical variables
    categoricals = ['weekday', 'hour', 'useragent', 'region', 'city', 'adexchange', 'slotwidth',
                    'slotheight', 'slotvisibility', 'slotformat', 'advertiser']

    for tag in categoricals:
        s = pd.Series(data[tag])
        d = pd.get_dummies(s, dummy_na=True)
        
        for k in d.keys():
            data[tag + '_' + str(k)] = d[k]
        
        data = data.drop(tag, axis = 1)
    
    # split usertag using ','
    """new_tags = data['usertag'].str.split(',')
    new_tags = new_tags.str.join('|').str.get_dummies()
    new_tags = new_tags.add_prefix('usertag_')
    data = data.join(colums_split)"""

    data.fillna("unknown", inplace=True)
    data = pd.get_dummies(data)
    
    # match test set and training set columns
    if enforce_cols is not None:
    # enforce_cols is the columns of train set, to_drop and to_add finds the difference
        to_drop = np.setdiff1d(data.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, data.columns)
        data.drop(to_drop, axis=1, inplace=True)
        data = data.assign(**{c: 0 for c in to_add})
        
    data = data.reindex(sorted(data.columns), axis=1)
        
    return data


# data preprocessing
train = downsampling(train_set)
train = data_preprocessing(train)
valid = data_preprocessing(valid_set, train.columns)
test = data_preprocessing(test_set, train.columns)

to_drop_columns = ['bidprice', 'payprice']
train = train.drop(to_drop_columns, axis = 1)
valid = valid.drop(to_drop_columns, axis = 1)
test = test.drop(to_drop_columns, axis = 1)
test = test.drop('click', axis = 1)

# split to x and y
train_x = train.drop('click', axis = 1)
train_y = train['click']
valid_x = valid.drop('click', axis = 1)
valid_y = valid['click']

# Logistic regression pCTR prediction
error = sys.maxsize
best_model = None
pCTR = None
for i in range(5):
    # default L2 penalty
    model = linear_model.LogisticRegression()
    model.fit(train_x, train_y)
    # get probability of click = 1
    valid_p = model.predict_proba(valid_x)[:,1]
    rms = sqrt(mean_squared_error(valid_y, valid_p))
    if rms < error:
        error = rms
        best_model = model
        pCTR = valid_p


# bid estimation
avgCTR = sum(train_set['click']) / len(train_set)

result_base_bid = []
result_clicks = []
result_CTR = []
result_spend = []
result_aCPM = []
result_aCPC = []

for base_bid in range(20, 80, 1):
    #print(base_bid)
    bids = [p * base_bid / avgCTR for p in pCTR]
    
    clicks = 0
    winning_impressions = 0
    spend = 0
    budget = 6250 * 1000
    
    for i in range(len(valid_set)):
        if bids[i] > budget - spend:
            bid = budget - spend
        else:
            bid = bids[i]
        if bid >= valid_set['payprice'][i]:
            spend += valid_set['payprice'][i]
            winning_impressions += 1
            if str(valid_set['click'][i]) == '1':
                clicks += 1
    spend /= 1000
    
    if clicks == 0:
        aCPM = 0
        aCPC = 0
    else:
        aCPM = spend / winning_impressions * 1000
        aCPC = spend / clicks
    CTR = clicks / winning_impressions
    
    result_base_bid.append(base_bid)
    result_clicks.append(clicks)
    result_CTR.append(CTR)
    result_spend.append(spend)
    result_aCPM.append(aCPM)
    result_aCPC.append(aCPC)

# save file
submission = pd.DataFrame({'base_bid': result_base_bid, 'Clicks': result_clicks, 'CTR': result_CTR,
                           'Spend': result_spend, 'aCPM': result_aCPM, 'aCPC': result_aCPC})
submission.to_csv('Question3.csv', index=False)