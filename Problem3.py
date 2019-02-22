# int/int -> float
from __future__ import division


def getAvgCTRBaseBid():
    # read file
    f = open('train.csv', 'r')
    rows_train = f.readlines()[1:]
    f.close()
    # get number of clicks and impressions
    clicks = 0
    winning_impressions = 0
    payprices = 0
    for row in rows_train:
        winning_impressions += 1
        payprice = int(row.split(',')[21])
        payprices += int(payprice)
        if row.split(',')[0] == '1':
            clicks += 1

    # get avgCTR and base_bid
    avgCTR = clicks / winning_impressions
    base_bid = payprices / len(rows_train)
    print('\nclick_through_rate:\n' + str(avgCTR))
    print('\nbase_bid :\n' + str(base_bid))
    return (avgCTR, base_bid)


def linearBidding(avgCTR, base_bid):
    # evaluate on validation.csv
    f = open('validation.csv', 'r')
    rows_validation = f.readlines()[1:]
    f.close()
    clicks = 0
    winning_impressions = 0
    spend = 0
    pCTR = 0.00085
    bidprice = base_bid / avgCTR * pCTR;
    for row in rows_validation:
        payprice = int(row.split(',')[21])
        if bidprice >= payprice:
            spend += payprice / 1000
            winning_impressions += 1
            if row.split(',')[0] == '1':
                clicks += 1
            if spend > 6250:
                spend = 6250
                break

    click_through_rate = "{:.3%}".format(clicks / winning_impressions)

    if clicks == 0:
        average_cpm = 0
        average_cpc = 0
    else:
        average_cpm = spend / clicks * 1000
        average_cpc = spend / clicks

    print('\nclicks:\n' + str(clicks))
    print('\nclick_through_rate:\n' + str(click_through_rate))
    print('\nspend:\n' + str(spend))
    print('\naverage_cpm:\n' + str(average_cpm))
    print('\naverage_cpc:\n' + str(average_cpc))


avgCTR = getAvgCTRBaseBid()[0]
base_bid = getAvgCTRBaseBid()[1]
linearBidding(avgCTR, base_bid)

'''
# save bidprice for test.csv
f = open('test.csv', 'r')
rows_test = f.readlines()[1:]
f.close()
f = open('testing_bidding_price.csv', 'w')
f.write('bidid,bidprice\n')
for row in rows_test:
    bidid = row.split(',')[2]
    bidprice = constant_bidprice[row.split(',')[20]]
    f.write(bidid + ',' + str(bidprice) + '\n')
f.close()
'''

