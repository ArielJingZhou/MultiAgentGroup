# int/int -> float
from __future__ import division


def getbidprice():
    # read file
    f = open('train.csv', 'r')
    rows = f.readlines()[1:]
    f.close()
    # get payprice
    payprices = 0
    for row in rows:
        payprice = row.split(',')[21]
        payprices += int(payprice)

    # get constant_bidprice: mean of pay_price
    return payprices / len(rows)


def const(constant_bidprice):
    # evaluate on validation.csv
    f = open('validation.csv', 'r')
    rows_validation = f.readlines()[1:]
    f.close()
    clicks = 0
    winning_impressions = 0
    spend = 0
    for row in rows_validation:
        payprice = int(row.split(',')[21])
        if constant_bidprice >= payprice:
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
        average_cpm = spend / winning_impressions * 1000
        average_cpc = spend / clicks

    print('constant_bidprice:\n' + str(constant_bidprice))
    print('\nclicks:\n' + str(clicks))
    print('\nclick_through_rate:\n' + str(click_through_rate))
    print('\nspend:\n' + str(spend))
    print('\naverage_cpm:\n' + str(average_cpm))
    print('\naverage_cpc:\n' + str(average_cpc))


const(getbidprice());

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

