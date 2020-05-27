import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import sys

def buy(buy_price, money):
    hold = money / buy_price
    money = 0
    return hold, money

def sold(sold_price, hold):
    money = hold * sold_price
    hold = 0
    return hold, money


if __name__ == '__main__':
    price = pd.read_csv(sys.argv[1])
    price['Date'] = price['Date'].apply(lambda x:x.replace('-', ''))
    price = price.dropna()
    
    data = pd.read_csv(sys.argv[2])
    date = list([str(x) for x in data['date']])
    predict = list(data['predict'])
    
    money = 10000.0
    hold = 0
    count = 0
    fee = 0
    buy_time = []
    sell_time = []
    for i in range(len(date)):
        Adj = price[price['Date']==date[i]]['Adj Close'].values[0]
        if hold == 0:
            if predict[i] == 1:
                hold, money = buy(Adj, money)
                buy_time.append(i)
                count = 3
                buy_price = Adj
                fee = hold * buy_price * 0.001425

        else:
            count -= 1
            if count != 0 and (Adj / buy_price) > 1.001:

                if predict[i] == 1:
                    count = 3
                else:     
                    fee += hold * Adj * 0.001425
                    hold, money = sold(hold, Adj)
                    sell_time.append(i)
                    count = 0
                    money -= fee
                    fee = 0

            elif count == 0 and hold != 0:
                if predict[i] == 1:
                    count = 3
                else:
                    fee += hold * Adj * 0.001425
                    hold, money = sold(hold, Adj)
                    sell_time.append(i)
                    money -= fee
                    fee = 0               

#         print('Money: {:.4f}, hold: {:.4f}'.format(money, hold))

    if hold != 0:
        fee += hold * Adj * 0.001425
        hold, money = sold(hold, Adj)
        sell_time.append(i)
        money -= fee
        fee = 0

    print('Return: ', money/10000) 
    
    
#     price = price[price['Date']<= date[-1]]
#     price = price[price['Date']>= date[0]]

#     fig = plt.figure(figsize=(12, 8))

#     days = [datetime.strptime(x, '%Y%m%d') for x in list(price['Date'])]

#     ax1=fig.add_subplot(1,1,1)

#     ax1.plot(days, list(price['Adj Close']), c='gray')
#     plt.gcf().autofmt_xdate()

#     alldays =  mdates.MonthLocator()
#     ax1.xaxis.set_major_locator(alldays)        
#     ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d')) 


#     ax1.scatter([days[x] for x in buy_time], [price.iloc[x]['Adj Close'] for x in buy_time], s=80, c = 'tab:blue', label='Buy')
#     ax1.scatter([days[x] for x in sell_time], [price.iloc[x]['Adj Close'] for x in sell_time], s=80, c = 'tab:cyan', label='Sell')
#     plt.xlabel('Date')
#     plt.ylabel('Index')

#     plt.legend()
#     plt.show()
