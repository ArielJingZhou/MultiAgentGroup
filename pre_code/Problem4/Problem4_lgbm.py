import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
from math import sqrt
from sklearn.metrics import mean_squared_error

print('Loading data...')
# load or create your dataset
train_set = pd.read_csv("train.csv")
valid_set = pd.read_csv("validation.csv")


test_set = pd.read_csv("test.csv")

# downsample train set, too big
def downsampling(data):
    no_click = data.query('click == 0')
    do_click = data.query('click == 1')
    nums = len(do_click) * 700
    new_no_click = no_click.sample(n=nums, random_state=42)
    return pd.concat([new_no_click, do_click])


# drop user tag for problem 3 ... to be considered more
# slotprice is considered as continuous variable

def data_preprocessing(data, enforce_cols=None):
    # data = data.sort_index(axis=0)

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

        data = data.drop(tag, axis=1)

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
train = train.drop(to_drop_columns, axis=1)
valid = valid.drop(to_drop_columns, axis=1)
test = test.drop(to_drop_columns, axis=1)
test = test.drop('click', axis=1)

# split to x and y
train_x = train.drop('click', axis=1)
train_y = train['click']
valid_x = valid.drop('click', axis=1)
valid_y = valid['click']

lgb_train = lgb.Dataset(train_x, train_y)
lgb_valid = lgb.Dataset(valid_x, valid_y)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

def predict_CTR():

    gbm = lgb.train(params,
                    lgb_train,
                    valid_sets=lgb_valid,
                    num_boost_round=20,
                    early_stopping_rounds=100)
    '''
    #save model
    model_file = open('../models/lgbm_model.sav', "wb")
    pickle.dump(gbm, model_file)
    model_file.close()
    
    #load model
    model_file = open('../models/lgbm_model.sav', 'rb')
    gbm = pickle.load(model_file)
    '''
    print('Starting prediction...')
    y_pred = gbm.predict(train_x)
    return y_pred

def non_linear_bidding(c, l, pCTRs):
    # evaluate on validation.csv
    clicks = 0
    winning_impressions = 0
    spend = 0
    budget = 6250 * 1000

    #bid_prices = (c / l * pCTRs + c * c) ** 0.5 - c
    bid_prices = c * (((pCTRs + (c*c*l*l+pCTRs*pCTRs)**0.5)/c/l) ** (1.0/3)-(c*l/(pCTRs + (c*c*l*l+pCTRs*pCTRs)**0.5))**(1.0/3))

    i = 0
    for i in range(len(valid_set)):
        if bid_prices[i] > budget - spend:
            bid = budget - spend
        else:
            bid = bid_prices[i]
        if bid >= valid_set['payprice'][i]:
            spend += valid_set['payprice'][i]
            winning_impressions += 1
            if str(valid_set['click'][i]) == '1':
                clicks += 1
    spend /= 1000

    click_through_rate = "{:.3%}".format(clicks / winning_impressions)

    if clicks == 0:
        average_cpm = 0
        average_cpc = 0
    else:
        average_cpm = spend / winning_impressions * 1000
        average_cpc = spend / clicks
    print('\nclicks:' + str(clicks))
    print('\nclick_through_rate:' + str(click_through_rate))
    print('\nspend:' + str(spend))
    print('\naverage_cpm:' + str(average_cpm))
    print('\naverage_cpc:' + str(average_cpc))

    #f = open("lgb_tuning_results1.csv", "a+")
    #f.write(str(c) + "," + str(l) + "," + str(clicks) + "," + str(click_through_rate) + "," + str(spend/1000) + ","  + str(average_cpm) + "," + str(average_cpc) + "\n")
    #f.close()

    return str(c) + "," + str(l) + "," + str(clicks) + "," + str(click_through_rate) + "," + str(spend) + "," \
                   + str(average_cpm) + "," + str(average_cpc)


pCTRs = predict_CTR()

#file = open("lgb_tuning_results1.csv", "w")
#file.write("constant, lambda, clicks, CTR, spend, CPM, CPC\n")
#file.close()

#for c in range(50, 100, 5):
#    for l in range(50, 70, 1):
#        l = l * 10 ** (-7)

#formula 1
#non_linear_bidding(75, 5.1 * 10 ** (-6), pCTRs)

#formula 2
non_linear_bidding(85, 3 * 10 ** (-6), pCTRs)