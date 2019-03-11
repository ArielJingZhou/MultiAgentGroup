# int/int -> float
from __future__ import division
import time
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import Pool, CatBoostRegressor

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
            X_dict.append([row[1], row[2], row[7],  row[8], row[14], row[15], row[17], row[18], row[23]])
        print(X_dict[0])
        return X_dict, y
    if data_type == 'validation':
        for row in rows_validation:
            row = row.split(',')
            X_dict.append([row[1], row[2], row[7],  row[8], row[14], row[15], row[17], row[18], row[23]])
        print(X_dict[0])
        return X_dict
    return 0


def model(load_model, model_type, bidding_function):
    # Create training and testing set
    X_dict_train, y_train = process_data('train')
    X_dict_validation = process_data('validation')

    # Load Model
    if load_model:
        print('Loading model from previous training...')
        if model_type == 'decision_tree':
            d_tree_file = open('./models/decision_tree_model.sav', 'rb')
        elif model_type == 'catboost':
            d_tree_file = open('./models/catboost_model.sav', 'rb')
        else:
            print("Cannot load model without model_type")
            return 0
        train_model = pickle.load(d_tree_file)
        d_tree_file.close()

    if load_model == False:
        # Transform training dictionary into one-hot encoded vectors
        print('Completed processing data')

        # Train decision tree classifier
        if model_type == 'decision_tree':
            train_model = DecisionTreeClassifier(criterion='gini', min_samples_split=30)
        elif model_type == 'catboost':
            train_pool = Pool(X_dict_train, y_train)
            # specify the training parameters
            train_model = CatBoostRegressor(iterations=2, depth=5, learning_rate=0.5, loss_function='RMSE')
            print("Started training...")
            train_model.fit(train_pool)
        else:
            print("Cannot set up model without model_type")
            return 0

        print('Completed training')


        # Save Model
        if model_type == 'decision_tree':
            model_file = open('./models/decision_tree_model.sav', "wb")
        elif model_type == 'catboost':
            model_file = open('./models/catboost_model.sav', "wb")
        else:
            print("Cannot save model without model_type")
            return 0
        pickle.dump(train_model, model_file)
        model_file.close()
        print('Saved model')

    # Evaluate and run model on validation data
    print('Tuning base bid for the model...')
    if model_type == 'decision_tree':
        f = open('tune_base_bid_decision_tree.csv', 'w')
    elif model_type == 'catboost':
        test_pool = Pool(X_dict_validation)
        pCTRs = train_model.predict(test_pool)
        f = open('tune_c_l_catboost.csv', 'w')
    else:
        print("Cannot save model without model_type")
        return 0
    f.write('c, lambda, clicks, CTR, spend, avgCPM, avgCPC\n')
    for c in range(50, 100, 2):
        for l in range(10, 50, 1):
            bidding_results = bidding(pCTRs, c, l*10**(-7), bidding_function)
            for bidding_result in bidding_results:
                f.write(str(bidding_result) + ',')
            f.write('\n')
    f.close()
    return 0


def bidding(pCTRs, c, l, bidding_function):
    if bidding_function == 1:
        bid_prices = (c / l * pCTRs + c * c) ** 0.5 - c
    elif bidding_function == 2:
        bid_prices = c * (((pCTRs + (c * c * l * l + pCTRs * pCTRs) ** 0.5) / c / l) ** (1.0 / 3) - (
                    c * l / (pCTRs + (c * c * l * l + pCTRs * pCTRs) ** 0.5)) ** (1.0 / 3))
    else:
        print("Cannot calculate bid_prices without model_type")
        return 0
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

    return [c, l, clicks, CTR, spend, avgCPM, avgCPC]


def main():
    start_time = time.time()
    '''
    # Decision Tree train 3 feature use 1202.7s
    print('LightGB:\n')
    # if train model, load_model=False
    model(load_model=False, model_type='decision_tree')
    '''
    # Catboost:training uses 1513.4s
    print('Catboost:\n')
    model(load_model=True, model_type='catboost', bidding_function=2)

    print('Completed in "\t\t{0:.1f}s'.format(time.time() - start_time))
    return 0


if __name__ == '__main__':
    main()
