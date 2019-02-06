# int/int -> float
from __future__ import division

# read file
f = open('train.csv', 'r')
rows = f.readlines()[1:]
f.close()
# get payprice
pay_prices = {}
for row in rows:
    advertiser = row.split(',')[23]
    payprice = row.split(',')[21]
    if advertiser not in pay_prices.keys():
        pay_prices[advertiser] = [int(payprice)]
    else:
        pay_prices[advertiser].append(int(payprice))
# get constant_bid_price: mean of pay_price
constant_bidprice = {}
for advertiser in pay_prices.keys():
    constant_bidprice[advertiser] = round(sum(pay_prices[advertiser]) / len(pay_prices[advertiser]))

# evaluate on validation.csv
f = open('validation.csv', 'r')
rows_validation = f.readlines()[1:]
f.close()
clicks = {}
winning_impressions = {}
for row in rows_validation:
    advertiser = row.split(',')[23]
    pay_price = row.split(',')[21]
    if advertiser not in clicks.keys():
        clicks[advertiser] = 0
        winning_impressions[advertiser] = 0
    if constant_bidprice[advertiser] > int(pay_price):
        winning_impressions[advertiser] += 1
        if row.split(',')[0] == '1':
            clicks[advertiser] += 1

click_through_rate = {}
for advertiser in clicks.keys():
    click_through_rate[advertiser] = "{:.3%}".format(clicks[advertiser] / winning_impressions[advertiser])
spend = {}
for advertiser in clicks.keys():
    spend[advertiser] = clicks[advertiser] * constant_bidprice[advertiser]
    if spend[advertiser] > 6250:
        spend[advertiser] = 6250
average_cpm = {}
for advertiser in constant_bidprice.keys():
    average_cpm[advertiser] = 1000 * constant_bidprice[advertiser]
average_cpc = constant_bidprice

print('constant_bidprice:\n' + str(constant_bidprice))
print('\nclicks:\n' + str(clicks))
print('\nclick_through_rate:\n' + str(click_through_rate))
print('\nspend:\n' + str(spend))
print('\naverage_cpm:\n' + str(average_cpm))
print('\naverage_cpc:\n' + str(average_cpc))

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

