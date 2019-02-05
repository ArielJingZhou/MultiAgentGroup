# int/int -> float
from __future__ import division

# read file
f = open('train.csv', 'r')
rows = f.readlines()[1:]
f.close()
# get pay_price
pay_price = {}
for i in rows:
    cur_advertiser = i.split(',')[23]
    cur_payprice = i.split(',')[21]
    if cur_advertiser not in pay_price.keys():
        pay_price[cur_advertiser] = [int(cur_payprice)]
    else:
        pay_price[cur_advertiser].append(int(cur_payprice))
# get constant_bid_price: mean of pay_price
constant_bidprice = {}
for i in pay_price.keys():
    constant_bidprice[i] = round(sum(pay_price[i])/len(pay_price[i]))

print(constant_bidprice)
# save bidprice for test.csv
f = open('test.csv', 'r')
rows_test = f.readlines()[1:]
f.close()
f = open('testing_bidding_price.csv', 'w')
f.write('bidid,bidprice\n')
for i in rows_test:
    bidid = i.split(',')[2]
    bidprice = constant_bidprice[i.split(',')[20]]
    f.write(bidid + ',' + str(bidprice) + '\n')
f.close()
