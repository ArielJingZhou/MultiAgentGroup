import random
import csv

def random_agents(budget, num_agents, bid_lower_bound, bid_upper_bound):
    # init agents profiles
    spend = [0] * num_agents
    num_imps = [0] * num_agents
    clicks = [0] * num_agents

    # evaluate on validation.csv
    f = open('data/validation.csv', 'r')
    rows_validation = f.readlines()[1:]
    f.close()
    
    # eval all bids
    for row in rows_validation:
        payprice = int(row.split(',')[21])
    
        largest_bid = 0
        second_bid = 0
        agent = -1
        
        # find largest bid, second largest bid and agent
        for i in range(num_agents):
            curr_bid = random.randint(bid_lower_bound, bid_upper_bound)
            if curr_bid > budget - spend[i]:
                curr_bid = budget - spend[i]
            # update second_bid
            if curr_bid == largest_bid:
                second_bid = largest_bid
                if random.randint(0, 1) == 1:
                    agent = i
            elif curr_bid > largest_bid:
                second_bid = largest_bid
                largest_bid = curr_bid
                agent = i
        
        # update agnets profiles
        if largest_bid >= payprice:
            spend[agent] += max(payprice, second_bid)
            num_imps[agent] += 1
            if row.split(',')[0] == '1':
                clicks[agent] += 1
                
    return spend, num_imps, clicks

# tune num agents
budget = 6750 * 1000
bid_lower_bound = 30
bid_upper_bound = 130

result = []

for num_agents in range(50, 101):
    total_spend, total_imps, total_clicks = random_agents(budget, num_agents, bid_lower_bound, bid_upper_bound)

    clicks = 0
    CTR = 0
    spend = 0
    avgCPM = 0
    avgCPC = 0
    imp = 0
    for i in range(num_agents):
        clicks += total_clicks[i]
        imp += total_imps[i]
        if total_imps[i] != 0:
            CTR += total_clicks[i] / total_imps[i]
        spend += total_spend[i]
        if total_clicks[i] != 0:
            total_spend[i] /= 1000
            avgCPM = total_spend[i] / imp * 1000
            avgCPC = total_spend[i] / total_clicks[i]
    market = [num_agents, bid_lower_bound, bid_upper_bound, clicks, CTR, spend / 1000, avgCPM, avgCPC]
    result.append(market)