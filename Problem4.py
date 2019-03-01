# int/int -> float
from __future__ import division
from catboost import Pool, CatBoostRegressor

f = open('train.csv', 'r')
rows_train = f.readlines()[1:]
f.close()
train_data = []
train_label = []
for row in rows_train:
    row_list = row.split(',')
    # weekday, hour, region, city, slotwidth, slotheight, slotformat, slotprice, advertiser
    train_data.append([row_list[1], row_list[2], row_list[7], row_list[8], row_list[14], row_list[15],
                       row_list[17], row_list[18], row_list[23]])
    train_label.append(row_list[0])
#print(train_label)
print(train_data[0])

f = open('validation.csv', 'r')
rows_validation = f.readlines()[1:]
f.close()
test_data = []
for row in rows_validation:
    row_list = row.split(',')
    test_data.append([row_list[1], row_list[2], row_list[7], row_list[8], row_list[14], row_list[15],
                      row_list[17], row_list[18], row_list[23]])
print(test_data[0])


def predict_CTR():
    # initialize Pool
    train_pool = Pool(train_data,
                      train_label)
    test_pool = Pool(test_data)

    # specify the training parameters
    model = CatBoostRegressor(iterations=2,
                              depth=5,
                              learning_rate=0.5,
                              loss_function='RMSE')
    # train the model
    model.fit(train_pool)
    # make the prediction using the resulting model
    preds = model.predict(test_pool)
    print(preds)
    return preds


def non_linear_bidding(c, l):
    # evaluate on validation.csv
    clicks = 0
    winning_impressions = 0
    spend = 0
    pCTRs = predict_CTR()
    bid_prices = (c / l * pCTRs + c * c) ** 0.5 - c
    print(bid_prices[0])
    i = 0
    for row in rows_validation:
        pay_price = int(row.split(',')[21])
        bid_price = bid_prices[i]
        if bid_price >= pay_price:
            spend += pay_price
            winning_impressions += 1
            if row.split(',')[0] == '1':
                clicks += 1
            if spend > 6250000:
                spend = 6250000
                break
        i += 1

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
    return str(c) + "," + str(l) + "," + str(clicks) + "," + str(click_through_rate) + "," + str(spend/1000) + "," \
           + str(average_cpm) + "," + str(average_cpc)


'''
tuning result: c = 80, l = 2.5 * 10 ** (-6) best
clicks	CTR	    spend	CPM	        CPC
90	    0.06%	6250	41.82084619	69.44444444

f = open("Problem4_tuning_results.csv", "w")
f.write("constant, lambda, clicks, CTR, spend, CPM, CPC\n")
for c in range(10, 85, 5):
    for l in range(1, 30, 1):
        print("\n\nc: " + str(c) + "l: " + str(l))
        f.write(non_linear_bidding(c, l * 10 ** (-7)) + "\n")
f.close()

'''
non_linear_bidding(80, 25 * 10 ** (-7))


