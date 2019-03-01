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
    train_data.append([row_list[1], row_list[2], row_list[7], row_list[8], row_list[14], row_list[15], row_list[17], row_list[18], row_list[23]])
    train_label.append(row_list[0])
#print(train_label)
print(train_data[0])

f = open('validation.csv', 'r')
#f = open('test.csv', 'r')
rows_validation = f.readlines()[1:]
f.close()
test_data = []
bid_ids = []
for row in rows_validation:
    row_list = row.split(',')
    test_data.append([row_list[1], row_list[2], row_list[7], row_list[8], row_list[14], row_list[15],
                      row_list[17], row_list[18], row_list[23]])
    #test_data.append([row_list[0], row_list[1], row_list[6], row_list[7], row_list[13], row_list[14], row_list[16], row_list[17], row_list[20]])
    #bid_ids.append(row_list[2])
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
    pCTRs = predict_CTR()
    # tune c and l
    #f = open("Problem4_tuning_results2.csv", "w")
    #f.write("constant, lambda, clicks, CTR, spend, CPM, CPC\n")
    #for c in range(30, 70, 1):
        #for l in range(20, 50, 1):
    clicks = 0
    winning_impressions = 0
    spend = 0
    #l = l * 10 ** (-7)
    bid_prices = (c / l * pCTRs + c * c) ** 0.5 - c
    #bid_prices = c * (((pCTRs + (c*c*l*l+pCTRs*pCTRs)**0.5)/c/l) ** (1.0/3)-(c*l/(pCTRs + (c*c*l*l+pCTRs*pCTRs)**0.5))**(1.0/3))

    '''
    f = open('testing_bidding_price.csv', 'w')
    f.write('bidid,bidprice\n')
    for i in range(len(bid_ids)):
        bid_id = bid_ids[i]
        bid_price = bid_prices[i]
        print(bid_prices[i])
        f.write(bid_id + ',' + str(bid_price) + '\n')
    f.close()
    '''
    i = 0
    for row in rows_validation:
        pay_price = int(row.split(',')[21])
        bid_price = bid_prices[i]
        if bid_price >= pay_price:
            spend += pay_price / 1000
            winning_impressions += 1
            if row.split(',')[0] == '1':
                clicks += 1
            if spend > 6250:
                spend = 6250
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
            #f.write(str(c) + "," + str(l) + "," + str(clicks) + "," + str(click_through_rate) + "," + str(spend/1000) + ","  + str(average_cpm) + "," + str(average_cpc) + "\n")

    return str(c) + "," + str(l) + "," + str(clicks) + "," + str(click_through_rate) + "," + str(spend) + "," \
                   + str(average_cpm) + "," + str(average_cpc)


non_linear_bidding(80, 25 * 10 ** (-7))

'''
tuning result: 
function 1:
c = 80, l = 2.5 * 10 ** (-6) best
clicks	CTR	    spend	CPM	        CPC
90	    0.06%	6250	41.82084619	69.44444444

function 2:
constant	lambda	    clicks	 CTR	 spend	 CPM	        CPC
60	        3.30E-06	86	     0.06%	 6250	 41541.21221	72674.4186

'''



