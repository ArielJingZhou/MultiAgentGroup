# int/int -> float
from __future__ import division
import csv
from random import randint

clicks = 0
winning_impressions = 0
spend = 0

results = []
with open("train.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        results.append(row[21])
del results[0]

pay_prices = []
for price in results:
    pay_prices.append(int(price))

upper_bound = max(pay_prices)
lower_bound = min(pay_prices)
random_bidprice = randint(lower_bound,upper_bound)

# evaluate on validation.csv
f = open('validation.csv', 'r')
rows_validation = f.readlines()[1:]
f.close()
clicks = 0
winning_impressions = 0
spend = 0
for row in rows_validation:
    payprice = int(row.split(',')[21])
    if random_bidprice > payprice:
        spend += payprice
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

print('Random_bidprice:\n' + str(random_bidprice))
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

