# Author: Doug Hart
# Title: VAR creation
# Project: Economic Forecasting
# Date Created: 1/9/2021
# Last Updated: 1/9/2021

import pandas as pd
import datetime

import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.datetools import dates_from_str

# Useful for viewing all columns in notebook
pd.set_option('display.max_columns', 40)


df = pd.read_pickle('cleaned2_121.pkl', compression='zip')
mdata = df[['pcgdp','mancap','unem','pctot','pcbusinv','pcC','pcI','pcipi','pcsp500']]


'''~~~~~~formatting index to work properly, may not be necessary~~~~~~'''

dates = df[['year', 'month']].astype(int).astype(str)
dates.reset_index(inplace=True,drop=True)
monthly = dates['year'] + "M" + dates['month']
monthly = dates_from_str(monthly)
mdata.index = pd.DatetimeIndex(monthly)

'''~~~~~~~~~~~~~~~~~~~~~~~~~~Model Creation~~~~~~~~~~~~~~~~~~~~~~~~~~'''
maw = VAR(mdata, freq='m')

# Traditionally with 6 lags:
results = maw.fit(6)
# Or to autoselect lag order based on info criterion:
results = maw.fit(maxlags=12, ic='bic')

# Note that we have to specify the “initial value” for the forecast:
lag_order = results.k_ar # this equals the number of lags used to build model

# To gen forecast of six steps out:
results.forecast(mdata.values[-lag_order:], 5)

# provides forecast and intervals 
results.forecast_interval(mdata.values[-lag_order:], 6)

# So to save as arrays:
point_fcast = results.forecast_interval(mdata.values[-lag_order:], 5)[0]
lower_bounds = results.forecast_interval(mdata.values[-lag_order:], 5)[1]
upper_bounds = results.forecast_interval(mdata.values[-lag_order:], 5)[2]

# Can plot with:
results.plot_forecast(6)
# THIS IS NOT A GOOD LOOKING GRAPH!
# Will want to implement own custom graphing function for variables of interest
# like with housing forecasting Deep Learning model

def get_calibration_data(df, n_results):
    '''
    Runs VAR code and saves resulting predictions to lists, then drops most recent row of data and repeats n_results 
    times. Coded for save calibration data for 6 period forecast. 
    inputs:
        df: dataframe containing data for use in VAR model
        n_results: int, number of periods backwards to run the model and retain results, must be greater than 6
    
    '''
    dfc = pd.DataFrame(df,copy=True)
    fm1 = []
    fm2 = []
    fm3 = []
    fm4 = []
    fm5 = []
    fm6 = []

    um1 = []
    um2 = []
    um3 = []
    um4 = []
    um5 = []
    um6 = []

    lm1 = []
    lm2 = []
    lm3 = []
    lm4 = []
    lm5 = []
    lm6 = []
    
    for i in range((n_results+5)):
        dfc.drop(dfc.tail(1).index,inplace=True)
        maw = VAR(dfc, freq='m')

        fm1.insert(0,point_fcast[0][0])
        fm2.insert(0,point_fcast[1][0])
        fm3.insert(0,point_fcast[2][0])
        fm4.insert(0,point_fcast[3][0])
        fm5.insert(0,point_fcast[4][0])
        fm6.insert(0,point_fcast[5][0])

        lm1.insert(0,point_fcast[0][1])
        lm2.insert(0,point_fcast[1][1])
        lm3.insert(0,point_fcast[2][1])
        lm4.insert(0,point_fcast[3][1])
        lm5.insert(0,point_fcast[4][1])
        lm6.insert(0,point_fcast[5][1])


        um1.insert(0,point_fcast[0][2])
        um2.insert(0,point_fcast[1][2])
        um3.insert(0,point_fcast[2][2])
        um4.insert(0,point_fcast[3][2])
        um5.insert(0,point_fcast[4][2])
        um6.insert(0,point_fcast[5][2])
    # Then to trim lists to proper intervals
    fm1 = fm1[:-5]
    fm2 = fm2[1:-4]
    fm3 = fm3[2:-3]
    fm4 = fm4[3:-2]
    fm5 = fm5[4:-2]
    fm6 = fm6[5:]

    um1 = um1[:-5]
    um2 = um2[1:-4]
    um3 = um3[2:-3]
    um4 = um4[3:-2]
    um5 = um5[4:-1]
    um6 = um6[5:]

    lm1 = lm1[:-5]
    lm2 = lm2[1:-4]
    lm3 = lm3[2:-3]
    lm4 = lm4[3:-2]
    lm5 = lm5[4:-1]
    lm6 = lm6[5:]
    
    out = pd.DataFrame((list(zip(fm1, fm2, fm3, fm4, fm5, fm6, lm1, lm2, lm3, lm4, lm5, lm6, um1, um2, um3, um4, um5, um6))), 
               columns =['p1p', 'p2p', 'p3p', 'p4p', 'p5p', 'p6p', 'p1l', 'p2l', 'p3l', 'p4l', 'p5l', 'p6l', 'p1u', 'p2u', 'p3u', 'p4u', 'p5u', 'p6u'],index=df.index[-n_results:])
    out['actual'] = df['pcgdp'][-n_results:]
    return out

def calibration_check(actual,upper,lower, bias_as_percent=False):
    '''
    To find how frequently true value falls within projected range
    Inputs:
        actual: historic values for comparison, series
        upper: upper bounds of interval forecast, series or list
        lower: lower bounds of interval forecast, series or list
    Outputs:
        calibration: count of times actual falls within interval divided by n
        bias: Indicates direction of bias in the event that the predictions are 
            consistently high or low.
            Returns times number of times actual exceeded upper bounds less the
            number of times actual fell below lower bounds. If equal to zero, any errors 
            are equally spread above and below target range. If bias_as_percent set to True;
            bias will be reported as a ratio to n, or the number of predicted values for comparison. 
    Options:
        bias_as_percent: Defaults to False, if true bias will be reported as ratio of bias to n 
    '''
    n = len(actual)
    count = 0
    bias = 0
    for i in range(len(actual)):
        if upper[i] >= actual[i] >= lower[i]:
            count += 1
        elif upper[i] < actual[i]:
            bias += 1
        elif actual[i] < lower[i]:
            bias -= 1
    calibration = count / n
    if bias_as_percent == True:
        return calibration, bias/n
    else:
        return calibration, bias



