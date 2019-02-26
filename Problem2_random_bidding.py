from __future__ import division
import csv
from random import randint

clicks = 0
winning_impressions = 0
spend = 0

def randbid():
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
    return random_bidprice

def evaluate(random_bidprice):
    f = open('validation.csv', 'r')
    rows_validation = f.readlines()[1:]
    f.close()
    clicks = 0
    winning_impressions = 0
    spend = 0
    for row in rows_validation:
        payprice = int(row.split(',')[21])
        if random_bidprice > payprice:
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

    printResults(random_bidprice, clicks, click_through_rate, spend, average_cpm, average_cpc)

def printResults(rnd_bidprice, clicks, ctr, spend, avg_cpm, avg_cpc):
    print('Random Bid Price:' + str(rnd_bidprice))
    print('Clicks:' + str(clicks))
    print('Click Through Rate:' + str(ctr))
    print('Spend: ' + str(spend))
    print('Average CPM:' + str(avg_cpm))
    print('Average CPC: ' + str(avg_cpc))

evaluate(randbid())
