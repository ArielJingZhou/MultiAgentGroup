from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

f = open('train.csv', 'r')
rows_train = f.readlines()[1:]
f.close()

#f = open('test.csv', 'r')
f = open('validation.csv', 'r')
rows_test = f.readlines()[1:]
f.close()


def process_data(data_type):
    X_dict = []
    if data_type == 'train':
        y = []
        for row in rows_train:
            row = row.split(',')
            y.append(int(row[0]))
            # test.csv:
            #X_dict.append({'useragent': row[4], 'slotprice': row[17], 'advertiser': row[20]})
            # validation.csv':
            X_dict.append({'useragent': row[5], 'slotprice': row[18], 'advertiser': row[23]})
        return X_dict, y
    if data_type == 'test':
        for row in rows_test:
            # validation.csv:
            row = row.split(',')
            # test.csv:
            #X_dict.append({'useragent': row[4], 'slotprice': row[17], 'advertiser': row[20]})
            # validation.csv':
            X_dict.append({'useragent': row[5], 'slotprice': row[18], 'advertiser': row[23]})
        return X_dict
    print(X_dict[0])
    return 0

# Create training and testing set
X_dict_train, y_train = process_data('train')
X_dict_test = process_data('test')

# Creating test set and turn into one-hot encoded vectors
dict_one_hot_encoder = DictVectorizer(sparse=False)
dict_one_hot_encoder.fit(X_dict_train)
X_train = dict_one_hot_encoder.transform(X_dict_train)
X_test = dict_one_hot_encoder.transform(X_dict_test)

# decision tree:
#train_model = DecisionTreeClassifier(criterion='gini', min_samples_split=30)
# random forest:
train_model = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30, n_jobs=-1)
train_model.fit(X_train, y_train)

pCTRs = train_model.predict_proba(X_test)[:, 1]

# decision tree:
#f = open('testing_bidding_price_decision_tree_linear.csv', 'w')
#f = open('validation_bidding_price_decision_tree_linear.csv', 'w')
# random forest:
#f = open('testing_bidding_price_random_forest_linear.csv', 'w')
f = open('validation_bidding_price_random_forest_linear.csv', 'w')
f.write('bidid,bidprice\n')

# decicion tree:
#bid_prices = 77 * pCTRs / 1793 * 2430981
# random forest:
bid_prices = 76 * pCTRs / 1793 * 2430981

for i in range(0, len(rows_test), 1):
    row = rows_test[i].split(',')
    bid_id = row[3]
    slot_price = row[17]
    bid_price = bid_prices[i]
    f.write(bid_id + ',' + str(bid_price) + '\n')
f.close()
