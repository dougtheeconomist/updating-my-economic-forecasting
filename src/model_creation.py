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
    mdata = dfc[['pcgdp','mancap','unem','pctot','pcbusinv','pcC','pcI','pcipi','pcsp500']]

    dates = dfc[['year', 'month']].astype(int).astype(str)
    dates.reset_index(inplace=True,drop=True)
    monthly = dates['year'] + "M" + dates['month']
    monthly = dates_from_str(monthly)
    mdata.index = pd.DatetimeIndex(monthly)
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
        mdata.drop(mdata.tail(1).index,inplace=True)
        maw = VAR(mdata, freq='m')
        results = maw.fit(6)
        lag_order = results.k_ar
        
        point_fcast = results.forecast_interval(mdata.values[-lag_order:], 6)[0]
        lower_bounds = results.forecast_interval(mdata.values[-lag_order:], 6)[1]
        upper_bounds = results.forecast_interval(mdata.values[-lag_order:], 6)[2]
        
        fm1.insert(0,point_fcast[0][0])
        fm2.insert(0,point_fcast[1][0])
        fm3.insert(0,point_fcast[2][0])
        fm4.insert(0,point_fcast[3][0])
        fm5.insert(0,point_fcast[4][0])
        fm6.insert(0,point_fcast[5][0])

        lm1.insert(0,lower_bounds[0][0])
        lm2.insert(0,lower_bounds[1][0])
        lm3.insert(0,lower_bounds[2][0])
        lm4.insert(0,lower_bounds[3][0])
        lm5.insert(0,lower_bounds[4][0])
        lm6.insert(0,lower_bounds[5][0])


        um1.insert(0,upper_bounds[0][0])
        um2.insert(0,upper_bounds[1][0])
        um3.insert(0,upper_bounds[2][0])
        um4.insert(0,upper_bounds[3][0])
        um5.insert(0,upper_bounds[4][0])
        um6.insert(0,upper_bounds[5][0])
    # Then to trim lists to proper intervals
    fm1 = fm1[5:]
    fm2 = fm2[4:-1]
    fm3 = fm3[3:-2]
    fm4 = fm4[2:-3]
    fm5 = fm5[1:-4]
    fm6 = fm6[:-5]

    um1 = um1[5:]
    um2 = um2[4:-1]
    um3 = um3[3:-2]
    um4 = um4[2:-3]
    um5 = um5[1:-4]
    um6 = um6[:-5]

    lm1 = lm1[5:]
    lm2 = lm2[4:-1]
    lm3 = lm3[3:-2]
    lm4 = lm4[2:-3]
    lm5 = lm5[1:-4]
    lm6 = lm6[:-5]
    
    out = pd.DataFrame((list(zip(fm1, fm2, fm3, fm4, fm5, fm6, lm1, lm2, lm3, lm4, lm5, lm6, um1, um2, um3, um4, um5, um6))), 
               columns =['p1p', 'p2p', 'p3p', 'p4p', 'p5p', 'p6p', 'p1l', 'p2l', 'p3l', 'p4l', 'p5l', 'p6l', 'p1u', 'p2u', 'p3u', 'p4u', 'p5u', 'p6u'],index=df.index[-n_results:])
    out['actual'] = df['pcgdp'][-n_results:]
    return out

def calibration_check(actual, lower, upper, bias_as_percent=False):
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



def report_calibration(df,n):
    '''
    Combines get_calibration_data and calibration_check functions 
    for ease of use. Prints results for forecasts one to six months out
    inputs:
        df: dataframe containing data for modeling
        n: number of forecasts of past periods to generate for comparison
    '''
    report_list = get_calibration_data(df,n)
    print('Month One Calibration: ', calibration_check(report_list.actual, report_list.p1l, report_list.p1u))
    print('Month Two Calibration: ', calibration_check(report_list.actual, report_list.p2l, report_list.p2u))
    print('Month Three Calibration: ', calibration_check(report_list.actual, report_list.p3l, report_list.p3u))
    print('Month Four Calibration: ', calibration_check(report_list.actual, report_list.p4l, report_list.p4u))
    print('Month Five Calibration: ', calibration_check(report_list.actual, report_list.p5l, report_list.p5u))
    print('Month Six Calibration: ', calibration_check(report_list.actual, report_list.p6l, report_list.p6u))