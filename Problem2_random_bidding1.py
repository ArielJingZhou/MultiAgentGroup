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
random_bid = randint(lower_bound, upper_bound)


results2 = []
clicks_list = []
with open("validation.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        results2.append(row[21])
        clicks_list.append(row[0])
del results2[0]

pay_prices2 = []
for price in results2:
    pay_prices.append(int(price))

for p in pay_prices:
    if random_bid > p:
        spend += p
        winning_impressions +=1
        if spend > 6250:spend = 6250

for c in clicks_list:
    if c == '1':
        clicks += 1

click_through_rate = "{:.3%}".format(clicks / winning_impressions)

if clicks == 0:
    average_cpm = 0
    average_cpc = 0
else:
    average_cpm = spend / clicks * 1000
    average_cpc = spend / clicks

print('Random Bid Price:\n' + str(random_bid))
print('\nclicks:\n' + str(clicks))
print('\nclick_through_rate:\n' + str(click_through_rate))
print('\nspend:\n' + str(spend))
print('\naverage_cpm:\n' + str(average_cpm))
print('\naverage_cpc:\n' + str(average_cpc))