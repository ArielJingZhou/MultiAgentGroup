# int/int -> float
from __future__ import division
import time
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

f = open('train.csv', 'r')
rows_train = f.readlines()[1:]
f.close()

f = open('validation.csv', 'r')
rows_validation = f.readlines()[1:]
f.close()


def process_data(data_type):
    X_dict = []
    if data_type == 'train':
        y = []
        for row in rows_train:
            row = row.split(',')
            y.append(int(row[0]))
            X_dict.append({'useragent': row[5], 'slotprice': row[18], 'advertiser': row[23]})
        return X_dict, y
    if data_type == 'validation':
        for row in rows_validation:
            row = row.split(',')
            X_dict.append({'useragent': row[5], 'slotprice': row[18], 'advertiser': row[23]})
        return X_dict
    return 0



def model(load_model, model_type):
    # Create training and testing set
    X_dict_train, y_train = process_data('train')
    X_dict_validation = process_data('validation')

    # Creating test set and turn into one-hot encoded vectors
    dict_one_hot_encoder = DictVectorizer(sparse=False)
    dict_one_hot_encoder.fit(X_dict_train)
    X_validation = dict_one_hot_encoder.transform(X_dict_validation)

    # Load Model
    if load_model:
        print('Loading model from previous training...')
        if model_type == 'decision_tree':
            d_tree_file = open('./models/decision_tree_model.sav', 'rb')
        elif model_type == 'random_forest':
            d_tree_file = open('./models/random_forest_model.sav', 'rb')
        else:
            print("Cannot load model without model_type")
            return 0
        train_model = pickle.load(d_tree_file)
        d_tree_file.close()

    if load_model == False:
        # Transform training dictionary into one-hot encoded vectors
        X_train = dict_one_hot_encoder.transform(X_dict_train)
        print('Completed processing data')

        # Train decision tree classifier
        if model_type == 'decision_tree':
            train_model = DecisionTreeClassifier(criterion='gini', min_samples_split=30)
        elif model_type == 'random_forest':
            train_model = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=30, n_jobs=-1)
        else:
            print("Cannot set up model without model_type")
            return 0
        print("Started training...")
        train_model.fit(X_train, y_train)
        print('Completed training')


        # Save Model
        if model_type == 'decision_tree':
            model_file = open('./models/decision_tree_model.sav', "wb")
        elif model_type == 'random_forest':
            model_file = open('./models/random_forest_model.sav', "wb")
        else:
            print("Cannot save model without model_type")
            return 0
        pickle.dump(train_model, model_file)
        model_file.close()
        print('Saved model')

    # Evaluate and run model on validation data
    print('Tuning base bid for the model...')
    pCTRs = train_model.predict_proba(X_validation)[:, 1]
    if model_type == 'decision_tree':
        f = open('tune_base_bid_decision_tree.csv', 'w')
    elif model_type == 'random_forest':
        f = open('tune_base_bid_random_forest.csv', 'w')
    else:
        print("Cannot save model without model_type")
        return 0
    f.write('basebid,clicks, CTR, spend, avgCPM, avgCPC\n')
    for base_bid in range(1, 201, 1):
        bidding_results = bidding(pCTRs, base_bid)
        for bidding_result in bidding_results:
            f.write(str(bidding_result) + ',')
        f.write('\n')
    f.close()
    return 0


def bidding(pCTRs, base_bid):
    bid_prices = base_bid * pCTRs / 1793 * 2430981
    clicks = 0
    winning_impressions = 0
    spend = 0
    for i in range(0,len(rows_validation),1):
        row = rows_validation[i]
        pay_price = int(row.split(',')[21])
        if bid_prices[i] >= pay_price:
            spend += pay_price / 1000
            winning_impressions += 1
            if row.split(',')[0] == '1':
                clicks += 1
            if spend > 6250:
                spend = 6250
                break

    CTR = "{:.3%}".format(clicks / winning_impressions)
    print(winning_impressions)
    if clicks == 0:
        avgCPM = 0
        avgCPC = 0
    else:
        avgCPM = spend / winning_impressions * 1000
        avgCPC = spend / clicks

    return [base_bid, clicks, CTR, spend, avgCPM, avgCPC]


def main():
    start_time = time.time()

    # Decision Tree train 3 feature use 1165.7s
    print('Decision Tree:\n')
    # if train model, load_model=False
    model(load_model=False, model_type='decision_tree')
    '''
    # Random Forest: training uses 3333.3s
    print('Random Forest:\n')
    model(load_model=False, model_type='random_forest')
    '''
    print('Completed in "\t\t{0:.1f}s'.format(time.time() - start_time))
    return 0


if __name__ == '__main__':
    main()

'''
tune result:
decision tree: base_bid = 66
clicks	 CTR	 spend	 avgCPM	    avgCPC
76	     0.068%	 6250	 55.63	    82.24

random forest: base_bid = 66
clicks	 CTR	 spend	 avgCPM	    avgCPC
75	     0.067%	 6232.3	 55.55	    83.10

'''