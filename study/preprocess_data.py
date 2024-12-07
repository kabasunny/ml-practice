import pandas as pd

def preprocess_data(data):
    data['Date'] = data.index
    data['Week'] = data['Date'].dt.isocalendar().week
    data['Year'] = data['Date'].dt.isocalendar().year
    # print(data.head())
    '''
                  Open    High     Low   Close  ...    Volume       Date Week  Year
    Date                                        ...
    2023-12-01  2819.0  2842.0  2803.0  2833.0  ...  26774000 2023-12-01   48  2023     
    2023-12-04  2802.0  2802.5  2744.5  2767.5  ...  30495700 2023-12-04   49  2023     
    2023-12-05  2770.0  2784.5  2743.5  2753.5  ...  24512600 2023-12-05   49  2023     
    '''
    return data
