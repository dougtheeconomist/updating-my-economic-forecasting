# Author: Doug Hart
# Title: VAR creation
# Project: Economic Forecasting
# Date Created: 1/9/2021
# Last Updated: 2/16/2021

import pandas as pd
import numpy as np
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


def mape_calc(df,actual = str,forecast = str):
    '''
    Calculates Mean Absolute Percentage Error of forecasted data from actual values
    Args:
        df: dataframe containing columns with data of interest
        actual: column of actual target data for comparison, type = str
        forecast: column of predicted future values, type = str
    '''
    mape = np.sum(np.abs((df[actual] - df[forecast]) / df[actual])) / len(df[actual])
    return mape

def report_calibration(df,n,bp=False):
    '''
    Combines get_calibration_data and calibration_check functions 
    for ease of use. Prints results for forecasts one to six months out.
    inputs:
        df: dataframe containing data for modeling
        n: number of forecasts of past periods to generate for comparison
        bp: defaults to False, for use if variable of interest is first column of data.
            To evaluate by parts model, set to True. This will add first 4 columns to create
            GDP measure.
    '''
    if bp == False:
        report_list = get_calibration_data(df,n)
    elif bp == True:
        report_list = get_by_parts_calibration_data(df,n)
        
    print('Month One Calibration: ', calibration_check(report_list.actual, report_list.p1l, report_list.p1u),'MAPE: ',mape_calc(report_list,'actual','p1p'))
    print('Month Two Calibration: ', calibration_check(report_list.actual, report_list.p2l, report_list.p2u),'MAPE: ',mape_calc(report_list,'actual','p2p'))
    print('Month Three Calibration: ', calibration_check(report_list.actual, report_list.p3l, report_list.p3u),'MAPE: ',mape_calc(report_list,'actual','p3p'))
    print('Month Four Calibration: ', calibration_check(report_list.actual, report_list.p4l, report_list.p4u),'MAPE: ',mape_calc(report_list,'actual','p4p'))
    print('Month Five Calibration: ', calibration_check(report_list.actual, report_list.p5l, report_list.p5u),'MAPE: ',mape_calc(report_list,'actual','p5p'))
    print('Month Six Calibration: ', calibration_check(report_list.actual, report_list.p6l, report_list.p6u),'MAPE: ',mape_calc(report_list,'actual','p6p'))


'''
Future work: need to reformat these functions to be able to take number of 
periods to predict as an argument rather than be hardcoded to 6 periods out.
Probably a fun weekend project.

Less pressing functionality to add would be to allow selection of which variable
to calibrate rather than assuming column in X matrix. 

Once done could consider bundling as Class object.
'''


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~By Parts~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''



def get_by_parts_calibration_data(df, n_results):
    '''
    Runs VAR code and saves resulting predictions to lists, then drops most recent row of data and repeats n_results 
    times. Coded for save calibration data for 6 period forecast. 
    inputs:
        df: dataframe containing data for use in VAR model
        n_results: int, number of periods backwards to run the model and retain results, must be greater than 6
    
    '''
    def pc_convert(pre,post):
        return ((post - pre) / pre)*100
    
    dfc = pd.DataFrame(df,copy=True)
    bp_data = dfc[['C','I','G','net_exports','unem','meanprice','mancap','man_industelect','electtot']]
    dates = dfc[['year', 'month']].astype(int).astype(str)
    dates.reset_index(inplace=True,drop=True)
    monthly = dates['year'] + "M" + dates['month']
    monthly = dates_from_str(monthly)
    bp_data.index = pd.DatetimeIndex(monthly)
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
        bp_data.drop(bp_data.tail(1).index,inplace=True)
        b_p = VAR(bp_data, freq='m')
#         results = b_p.fit(maxlags=12, ic='bic')
        results = b_p.fit(6)
        lag_order = results.k_ar
        
        point_fcast = results.forecast_interval(bp_data.values[-lag_order:], 6)[0]
        lower_bounds = results.forecast_interval(bp_data.values[-lag_order:], 6)[1]
        upper_bounds = results.forecast_interval(bp_data.values[-lag_order:], 6)[2]
        
        # Stata code for example
        # pcforecast = ((forecast- L.forecast)/L.forecast)*100
        
        fm1.insert(0, pc_convert(np.sum(bp_data.iloc[-1][0:4]), np.sum(point_fcast[0][0:4])))
        fm2.insert(0, pc_convert(np.sum(point_fcast[0][0:4]), np.sum(point_fcast[1][0:4])))
        fm3.insert(0, pc_convert(np.sum(point_fcast[1][0:4]), np.sum(point_fcast[2][0:4])))
        fm4.insert(0, pc_convert(np.sum(point_fcast[2][0:4]), np.sum(point_fcast[3][0:4])))
        fm5.insert(0, pc_convert(np.sum(point_fcast[3][0:4]), np.sum(point_fcast[4][0:4])))
        fm6.insert(0, pc_convert(np.sum(point_fcast[4][0:4]), np.sum(point_fcast[5][0:4])))

        lm1.insert(0, pc_convert(np.sum(bp_data.iloc[-1][0:4]), np.sum(lower_bounds[0][0:4])))
        lm2.insert(0, pc_convert(np.sum(lower_bounds[0][0:4]), np.sum(lower_bounds[1][0:4])))
        lm3.insert(0, pc_convert(np.sum(lower_bounds[1][0:4]), np.sum(lower_bounds[2][0:4])))
        lm4.insert(0, pc_convert(np.sum(lower_bounds[2][0:4]), np.sum(lower_bounds[3][0:4])))
        lm5.insert(0, pc_convert(np.sum(lower_bounds[3][0:4]), np.sum(lower_bounds[4][0:4])))
        lm6.insert(0, pc_convert(np.sum(lower_bounds[4][0:4]), np.sum(lower_bounds[5][0:4])))


        um1.insert(0, pc_convert(np.sum(bp_data.iloc[-1][0:4]), np.sum(upper_bounds[0][0:4])))
        um2.insert(0, pc_convert(np.sum(upper_bounds[0][0:4]), np.sum(upper_bounds[1][0:4])))
        um3.insert(0, pc_convert(np.sum(upper_bounds[1][0:4]), np.sum(upper_bounds[2][0:4])))
        um4.insert(0, pc_convert(np.sum(upper_bounds[2][0:4]), np.sum(upper_bounds[3][0:4])))
        um5.insert(0, pc_convert(np.sum(upper_bounds[3][0:4]), np.sum(upper_bounds[4][0:4])))
        um6.insert(0, pc_convert(np.sum(upper_bounds[4][0:4]), np.sum(upper_bounds[5][0:4])))
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
    out['actual'] = dfc['pcgdp'][-n_results:]
    return out


# V2 actually seems to make worse interval predictions in first period, but more accurate
# predictins in subsequent periods. This result holds whether results fitted to six lags or 
# IC determined fit. Next step to check MAPE scores 
# to eval accuracy of point predictions. 
def get_by_parts_calibration_data_V2(df, n_results):
    '''
    Runs VAR code and saves resulting predictions to lists, then drops most recent row of data 
    and repeats n_results times. Coded for save calibration data for 6 period forecast.

    V2 forecasts relevant variables in percentage change form, then converts to actual values 
    to save for comparison. 
    inputs:
        df: dataframe containing data for use in VAR model
        n_results: int, number of periods backwards to run the model and retain results, must be greater than 6
    
    '''
    dfc = pd.DataFrame(df,copy=True)
    bp_data = dfc[['C','I','G','net_exports','pcC','pcI','pcG','pc_ne','unem','pc_mp','pc_mc','pc_mie','pc_et']]
    dates = dfc[['year', 'month']].astype(int).astype(str)
    dates.reset_index(inplace=True,drop=True)
    monthly = dates['year'] + "M" + dates['month']
    monthly = dates_from_str(monthly)
    bp_data.index = pd.DatetimeIndex(monthly)
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
    
    def pc_convert(pre,post):
        return ((post - pre) / pre)*100
    
    for i in range((n_results+5)):
        bp_data.drop(bp_data.tail(1).index,inplace=True)
        b_p = VAR(bp_data.iloc[:,4:], freq='m')
#         results = b_p.fit(maxlags=12, ic='bic')
        results = b_p.fit(6)
        lag_order = results.k_ar
        
        init = np.array([bp_data.iloc[-1][0:4]])
        point_fcast = results.forecast_interval(bp_data.values[-lag_order:,4:], 6)[0]
        lower_bounds = results.forecast_interval(bp_data.values[-lag_order:,4:], 6)[1]
        upper_bounds = results.forecast_interval(bp_data.values[-lag_order:,4:], 6)[2]
        
        # Point forecast
        (np.array([25,15,-10,75])/100)+1
        working_set = (np.array([point_fcast[0][0:4]])/100)+1
        new_bline = init * working_set
        fm1.insert(0,np.sum(new_bline))
        
        working_set = (np.array([point_fcast[1][0:4]])/100)+1
        new_bline = new_bline * working_set
        fm2.insert(0,np.sum(new_bline))
        
        working_set = (np.array([point_fcast[2][0:4]])/100)+1
        new_bline = new_bline * working_set
        fm3.insert(0,np.sum(new_bline))
        
        working_set = (np.array([point_fcast[3][0:4]])/100)+1
        new_bline = new_bline * working_set
        fm4.insert(0,np.sum(new_bline))
        
        working_set = (np.array([point_fcast[4][0:4]])/100)+1
        new_bline = new_bline * working_set
        fm5.insert(0,np.sum(new_bline))
        
        working_set = (np.array([point_fcast[5][0:4]])/100)+1
        new_bline = new_bline * working_set
        fm6.insert(0,np.sum(new_bline))
        
        # Lower interval bounds
        working_set = (np.array([lower_bounds[0][0:4]])/100)+1
        new_bline = init * working_set
        lm1.insert(0,np.sum(new_bline))
        
        working_set = (np.array([lower_bounds[1][0:4]])/100)+1
        new_bline = new_bline * working_set
        lm2.insert(0,np.sum(new_bline))
        
        working_set = (np.array([lower_bounds[2][0:4]])/100)+1
        new_bline = new_bline * working_set
        lm3.insert(0,np.sum(new_bline))
        
        working_set = (np.array([lower_bounds[3][0:4]])/100)+1
        new_bline = new_bline * working_set
        lm4.insert(0,np.sum(new_bline))
        
        working_set = (np.array([lower_bounds[4][0:4]])/100)+1
        new_bline = new_bline * working_set
        lm5.insert(0,np.sum(new_bline))
        
        working_set = (np.array([lower_bounds[5][0:4]])/100)+1
        new_bline = new_bline * working_set
        lm6.insert(0,np.sum(new_bline))
        
        # Upper interval bounds
        working_set = (np.array([upper_bounds[0][0:4]])/100)+1
        new_bline = init * working_set
        um1.insert(0,np.sum(new_bline))
        
        working_set = (np.array([upper_bounds[1][0:4]])/100)+1
        new_bline = new_bline * working_set
        um2.insert(0,np.sum(new_bline))
        
        working_set = (np.array([upper_bounds[2][0:4]])/100)+1
        new_bline = new_bline * working_set
        um3.insert(0,np.sum(new_bline))
        
        working_set = (np.array([upper_bounds[3][0:4]])/100)+1
        new_bline = new_bline * working_set
        um4.insert(0,np.sum(new_bline))
        
        working_set = (np.array([upper_bounds[4][0:4]])/100)+1
        new_bline = new_bline * working_set
        um5.insert(0,np.sum(new_bline))
        
        working_set = (np.array([upper_bounds[5][0:4]])/100)+1
        new_bline = new_bline * working_set
        um6.insert(0,np.sum(new_bline))
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
    out['actual'] = dfc['gdp'][-n_results:]
    return out

def report_calibration_V2(df,n,bp=False):
    '''
    Combines get_calibration_data and calibration_check functions 
    for ease of use. Prints results for forecasts one to six months out.
    inputs:
        df: dataframe containing data for modeling
        n: number of forecasts of past periods to generate for comparison
        bp: defaults to False, for use if variable of interest is first column of data.
            To evaluate by parts model, set to True. This will add first 4 columns to create
            GDP measure.
    '''
    if bp == False:
        report_list = get_calibration_data(df,n)
    elif bp == True:
        report_list = get_by_parts_calibration_data_V2(df,n)
        
    print('Month One Calibration: ', calibration_check(report_list.actual, report_list.p1l, report_list.p1u),'MAPE: ',mape_calc(report_list,'actual','p1p'))
    print('Month Two Calibration: ', calibration_check(report_list.actual, report_list.p2l, report_list.p2u),'MAPE: ',mape_calc(report_list,'actual','p2p'))
    print('Month Three Calibration: ', calibration_check(report_list.actual, report_list.p3l, report_list.p3u),'MAPE: ',mape_calc(report_list,'actual','p3p'))
    print('Month Four Calibration: ', calibration_check(report_list.actual, report_list.p4l, report_list.p4u),'MAPE: ',mape_calc(report_list,'actual','p4p'))
    print('Month Five Calibration: ', calibration_check(report_list.actual, report_list.p5l, report_list.p5u),'MAPE: ',mape_calc(report_list,'actual','p5p'))
    print('Month Six Calibration: ', calibration_check(report_list.actual, report_list.p6l, report_list.p6u),'MAPE: ',mape_calc(report_list,'actual','p6p'))
    

# Checking out which variables may be destabilizing model
def volatility(col=str,report = False):
    '''
    Divides standard deviation by mean of a given column in dataframe df
    arguments:
    col: name of column, string
    report: defaults to False, if set to True, prints out volatility statement
    '''
    volatility = df[col].std() / df[col].mean()
    if report == True:
        print(f"Volatility of {col} is {volatility}.")
    return volatility

for i in ['C','I','G','net_exports','unem','meanprice','mancap','man_industelect','electtot']:
    volatility(i, report=True)
# unem is the only standout for high volatility in this list
# others in V2 list of perc change versions
def mape_calc(df,actual = str,forecast = str):
    '''
    Calculates Mean Absolute Percentage Error of forecasted data from actual values
    Args:
        df: dataframe containing columns with data of interest
        actual: column of actual target data for comparison, type = str
        forecast: column of predicted future values, type = str
    '''
    mape = np.sum((df[actual] - df[forecast]) / df[actual]) / len(df[actual])
    return mape