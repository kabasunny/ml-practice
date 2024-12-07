import random

def random_daily_buy(data):
    random.seed(42)  # 再現性のためにシードを設定
    data['Buy'] = False
    # print(data.head())
    '''
                  Open    High     Low   Close  ...       Date  Week  Year    Buy
    Date                                        ...
    2023-12-01  2819.0  2842.0  2803.0  2833.0  ... 2023-12-01    48  2023  False       
    2023-12-04  2802.0  2802.5  2744.5  2767.5  ... 2023-12-04    49  2023  False       
    2023-12-05  2770.0  2784.5  2743.5  2753.5  ... 2023-12-05    49  2023  False       
    2023-12-06  2770.5  2829.5  2758.0  2827.0  ... 2023-12-06    49  2023  False       
    2023-12-07  2800.0  2812.0  2776.0  2794.5  ... 2023-12-07    49  2023  False 
    '''

    grouped = data.groupby(['Year', 'Week'])
    for name, group in grouped:
        if not group.empty:
            random_day = group.sample(n=1).index
            data.loc[random_day, 'Buy'] = True
    # print(data.head())
    '''
                  Open    High     Low   Close  ...       Date  Week  Year    Buy       
    Date                                        ...
    2023-12-01  2819.0  2842.0  2803.0  2833.0  ... 2023-12-01    48  2023   True       
    2023-12-04  2802.0  2802.5  2744.5  2767.5  ... 2023-12-04    49  2023  False       
    2023-12-05  2770.0  2784.5  2743.5  2753.5  ... 2023-12-05    49  2023  False       
    2023-12-06  2770.5  2829.5  2758.0  2827.0  ... 2023-12-06    49  2023   True       
    2023-12-07  2800.0  2812.0  2776.0  2794.5  ... 2023-12-07    49  2023  False       
    
    '''

    buy_signals = data['Buy'].apply(lambda x: 'Buy' if x else 'Hold').tolist()
    # print(buy_signals)
    '''
    ['Buy', 'Hold', 'Hold', 'Buy', 'Hold', 'Hold', 'Hold', 'Hold', 'Hold', 'Hold', 'Buy', 'Hold', 'Hold', 'Hold', 'Hold', 'Buy', 'Hold', 'Hold', 'Buy', 'Hold', 'Hold']     
    '''
    return buy_signals
