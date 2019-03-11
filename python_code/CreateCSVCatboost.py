# int/int -> float
from __future__ import division
from sklearn.feature_extraction import DictVectorizer
from catboost import Pool, CatBoostRegressor
import pickle


#f = open('test.csv', 'r')
f = open('validation.csv', 'r')
rows_test = f.readlines()[1:]
f.close()

def process_data():
    X_dict = []
    for row in rows_test:
        row = row.split(',')
        #test.csv:
        #X_dict.append([row[0], row[1], row[6],  row[7], row[13], row[14], row[16], row[17], row[20]])
        #validation.csv:
        X_dict.append([row[1], row[2], row[7], row[8], row[14], row[15], row[17], row[18], row[23]])
    print(X_dict[0])
    return X_dict


d_tree_file = open('./models/catboost_model.sav', 'rb')
train_model = pickle.load(d_tree_file)
d_tree_file.close()


X_dict_test = process_data()

test_pool = Pool(X_dict_test)
pCTRs = train_model.predict(test_pool)
# ORTB1:
#f = open('testing_bidding_price_catboost_ORTB1.csv', 'w')
#f = open('validation_bidding_price_catboost_ORTB1.csv', 'w')
# ORTB2:
#f = open('testing_bidding_price_catboost_ORTB2.csv', 'w')
f = open('validation_bidding_price_catboost_ORTB2.csv', 'w')
f.write('bidid,bidprice\n')

# ORTB1:
c = 112
l = 2.8 * 10 ** (-6)
#bid_prices = (c / l * pCTRs + c * c) ** 0.5 - c
# ORTB2:
c = 96
l = 4.2 * 10 ** (-6)
bid_prices = c * (((pCTRs + (c * c * l * l + pCTRs * pCTRs) ** 0.5) / c / l) ** (1.0 / 3) - (c * l / (pCTRs + (c * c * l * l + pCTRs * pCTRs) ** 0.5)) ** (1.0 / 3))

for i in range(0, len(rows_test), 1):
    row = rows_test[i].split(',')
    bid_id = row[3]
    slot_price = row[17]
    bid_price = bid_prices[i]
    f.write(bid_id + ',' + str(bid_price) + '\n')
f.close()
