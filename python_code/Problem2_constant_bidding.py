# int/int -> float
from __future__ import division
import time

f = open('validation.csv', 'r')
rows_validation = f.readlines()[1:]
f.close()

def getbidprice(bid_strategy_type):
    # read file
    f = open('train.csv', 'r')
    rows = f.readlines()[1:]
    f.close()
    # get payprice
    # constant bidding: average of pay price
    if bid_strategy_type == 'constant_bidding':
        pay_prices = 0
        for row in rows:
            pay_price = row.split(',')[21]
            pay_prices += int(pay_price)
        bid_price = pay_prices / len(rows)

    # get constant_bidprice: mean of pay_price
    return bid_price


def bidding(bid_price):
    clicks = 0
    winning_impressions = 0
    spend = 0
    for row in rows_validation:
        pay_price = int(row.split(',')[21])
        if bid_price >= pay_price:
            spend += pay_price / 1000
            winning_impressions += 1
            if row.split(',')[0] == '1':
                clicks += 1
            if spend > 6250:
                spend = 6250
                break

    CTR = "{:.3%}".format(clicks / winning_impressions)

    if clicks == 0:
        avgCPM = 0
        avgCPC = 0
    else:
        avgCPM = spend / winning_impressions * 1000
        avgCPC = spend / clicks

    return [bid_price, clicks, CTR, spend, avgCPM, avgCPC]


def main():
    start_time = time.time()
    # tune bid_price
    f = open('tune_constant_value.csv', 'w')
    f.write('bidprice,clicks, CTR, spend, avgCPM, avgCPC\n')
    for bid_price in range(1, 500, 1):
        bidding_results = bidding(bid_price)
        for bidding_result in bidding_results:
            f.write(str(bidding_result) + ',')
        f.write('\n')
    f.close()
    print('Completed in \t\t{0:.1f}s'.format(time.time() - start_time))

if __name__ == '__main__':
    main()

'''
tune result:
constant bidding: bidprice = 79
clicks	 CTR	 spend	 avgCPM	    avgCPC
68	     0.05%	 6250	 42.8325692	91.91176471
'''