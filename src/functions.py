# Author: Doug Hart
# Title: VAR functions
# Project: Economic Forecasting
# Date Created: 4/8/2021
# Last Updated: 4/9/2021

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.datetools import dates_from_str
import matplotlib.pyplot as plt
# %matplotlib inline # if run in Jupyter


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Calibration and reporting functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

# As-a-whole model calibration backtesting data collection
def get_calibration_data(df, n_results):
    '''
    Runs VAR code and saves resulting predictions to lists, then drops most recent row of data and repeats n_results 
    times. Coded for save calibration data for 6 period forecast. 

    Parameters
    ----------
    df: dataframe containing data for use in VAR model
    n_results: int, number of periods backwards to run the model and retain results, must be greater than 6
    
    Returns
    -------
    out : pandas dataframe containing forecasting results from backtesting as well as target 
        variable for comparison. 

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
        results = maw.fit(maxlags=12, ic='bic')
        # results = maw.fit(6)
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

# By-parts model calibration backtesting data collection
def get_by_parts_calibration_data(df, n_results):
    '''
    Runs VAR code and saves resulting predictions to lists, then drops most recent row of data and repeats n_results 
    times. Coded for save calibration data for 6 period forecast. 

    Parameters
    ----------
    df: dataframe containing data for use in VAR model
    n_results: int, number of periods backwards to run the model and retain results, must be greater than 6
    
    Returns
    -------
    out : pandas dataframe containing forecasting results from backtesting as well as target 
        variable for comparison. 

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
        results = b_p.fit(maxlags=12, ic='bic')
        # results = b_p.fit(6)
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

def calibration_check(actual, lower, upper, bias_as_percent=False):
    '''
    To find how frequently true value falls within projected range

    Parameters
    ----------
    actual: historic values for comparison, series
    upper: upper bounds of interval forecast, series or list
    lower: lower bounds of interval forecast, series or list
    bias_as_percent: Defaults to False, if true bias will be reported as ratio of bias to n

    Returns
    -------
    out : calibration: count of times actual falls within interval divided by n
        bias: Indicates direction of bias in the event that the predictions are 
            consistently high or low.
            Returns times number of times actual exceeded upper bounds less the
            number of times actual fell below lower bounds. If equal to zero, any errors 
            are equally spread above and below target range. If bias_as_percent set to True;
            bias will be reported as a ratio to n, or the number of predicted values for comparison. 

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

    Parameters
    ----------
    df: dataframe containing columns with data of interest
    actual: column of actual target data for comparison, type = str
    forecast: column of predicted future values, type = str

    Returns
    -------
    out : error metric, float

    '''
    mape = np.sum(np.abs((df[actual] - df[forecast]) / df[actual])) / len(df[actual])
    return mape

def report_calibration(df,n,bp=False):
    '''
    
    Combines get_calibration_data and calibration_check functions 
    for ease of use. Prints results for forecasts one to six months out.

    Parameters
    ----------
    df: dataframe containing data for modeling
    n: number of forecasts of past periods to generate for comparison
    bp: defaults to False, for use if variable of interest is first column of data.
        To evaluate by parts model, set to True. This will sum first 4 columns to create
        GDP measure.

    Returns
    -------
    prints calibration and error score for each period forecast

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


# Checking out which variables may be destabilizing model
def volatility(df, col=str, report = False):
    '''
    Divides standard deviation by mean of a given column in dataframe df
    Also known as coefficient of variation

    Parameters
    ----------
    df: dataframe containing relevant data
    col: name of column within df, string
    report: defaults to False, if set to True, prints out volatility statement

    Returns
    -------
    out : float, volatility of given column

    '''
    volatility = df[col].std() / df[col].mean()
    if report == True:
        print(f"Volatility of {col} is {volatility}.")
    return volatility


'''
Future work: need to reformat these functions to be able to take number of 
periods to predict as an argument rather than be hardcoded to 6 periods out.
Probably a fun weekend project.

Less pressing functionality to add would be to allow selection of which variable
to calibrate rather than assuming column in X matrix. 

Once done could consider bundling as Class object.
'''

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Generation of Calibrated Forecasts~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

# Generating As-a-Whole forecast
def gen_forecast_w(df):
    '''
    Generates calibrated 6 period forecast of GPD using VAR model. 
    Uses GDP as-a-whole approach to predict GDP directly. 

    Parameters
    ----------
     df = dataframe with relevant data

    Returns
    -------
    Out : 6 period point forecast, 6 period lower interval, 6 period upper interval
        format =  6x1 numpy arrays
    
    '''
    dfc = pd.DataFrame(df,copy=True)
    mdata = dfc[['pcgdp','mancap','unem','pctot','pcbusinv','pcC','pcI','pcipi','pcsp500']]

    dates = dfc[['year', 'month']].astype(int).astype(str)
    dates.reset_index(inplace=True,drop=True)
    monthly = dates['year'] + "M" + dates['month']
    monthly = dates_from_str(monthly)
    mdata.index = pd.DatetimeIndex(monthly)
    maw = VAR(mdata, freq='m')
    results = maw.fit(maxlags=12, ic='bic')
    lag_order = results.k_ar

    w_correction = np.array([(.95 / .9), (.95 / .9), (.95 / .9), (.95 / .9), (.95 / .9), (.95 / .9)])
    
    w_point_fcast = results.forecast_interval(mdata.values[-lag_order:], 6)[0]
    w_lower_bounds = results.forecast_interval(mdata.values[-lag_order:], 6)[1]
    w_upper_bounds = results.forecast_interval(mdata.values[-lag_order:], 6)[2]
    
    w_pf = np.array([w_point_fcast[0][0], w_point_fcast[1][0], w_point_fcast[2][0], w_point_fcast[3][0], w_point_fcast[4][0], w_point_fcast[5][0]])
    w_lib = np.array([w_lower_bounds[0][0], w_lower_bounds[1][0], w_lower_bounds[2][0], w_lower_bounds[3][0], w_lower_bounds[4][0], w_lower_bounds[5][0]])
    w_uib = np.array([w_upper_bounds[0][0], w_upper_bounds[1][0], w_upper_bounds[2][0], w_upper_bounds[3][0], w_upper_bounds[4][0], w_upper_bounds[5][0]])
    
    w_adjustment = np.array(w_uib - w_lib)*w_correction - (w_uib - w_lib)
    w_adjustment = w_adjustment / 2
    w_lib = w_lib - w_adjustment
    w_uib = w_uib + w_adjustment
    
    return w_pf, w_lib, w_uib
    

# Generating By-Parts forecast
def gen_forecast_bp(df):
    '''
    Generates calibrated 6 period forecast of GPD using VAR model. 
    Uses GDP by-part approach to predict GDP by aggregation of component parts
    according to the equation GDP = C + I + G + net exports. 

    Parameters
    ---------- 
    df = dataframe with relevant data, percentage = defaults to True, setting
        to False toggles output from percentage change format to direct values
    
    Returns
    -------
    Out : 6 period point forecast, 6 period lower interval, 6 period upper interval
        output format is 6X1 numpy arrays
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

    b_p = VAR(bp_data, freq='m')
    bp_results = b_p.fit(maxlags=12, ic='bic')
    lag_order = bp_results.k_ar
    bp_correction = np.array([(.95 / .91), (.95 / .88), (.95 / .87), (.95 / .87), (.95 / .85), (.95 / .83)])

    point_fcast = bp_results.forecast_interval(bp_data.values[-lag_order:], 6)[0]
    lower_bounds = bp_results.forecast_interval(bp_data.values[-lag_order:], 6)[1]
    upper_bounds = bp_results.forecast_interval(bp_data.values[-lag_order:], 6)[2]

    # Aggregate to Point Forecast
    bp_pf = []
    bp_lib = []
    bp_uib = []

    bp_pf.append(pc_convert(np.sum(bp_data.iloc[-1][0:4]), np.sum(point_fcast[0][0:4])))
    bp_pf.append(pc_convert(np.sum(point_fcast[0][0:4]), np.sum(point_fcast[1][0:4])))
    bp_pf.append(pc_convert(np.sum(point_fcast[1][0:4]), np.sum(point_fcast[2][0:4])))
    bp_pf.append(pc_convert(np.sum(point_fcast[2][0:4]), np.sum(point_fcast[3][0:4])))
    bp_pf.append(pc_convert(np.sum(point_fcast[3][0:4]), np.sum(point_fcast[4][0:4])))
    bp_pf.append(pc_convert(np.sum(point_fcast[4][0:4]), np.sum(point_fcast[5][0:4])))

    bp_lib.append(pc_convert(np.sum(bp_data.iloc[-1][0:4]), np.sum(lower_bounds[0][0:4])))
    bp_lib.append(pc_convert(np.sum(lower_bounds[0][0:4]), np.sum(lower_bounds[1][0:4])))
    bp_lib.append(pc_convert(np.sum(lower_bounds[1][0:4]), np.sum(lower_bounds[2][0:4])))
    bp_lib.append(pc_convert(np.sum(lower_bounds[2][0:4]), np.sum(lower_bounds[3][0:4])))
    bp_lib.append(pc_convert(np.sum(lower_bounds[3][0:4]), np.sum(lower_bounds[4][0:4])))
    bp_lib.append(pc_convert(np.sum(lower_bounds[4][0:4]), np.sum(lower_bounds[5][0:4])))

    bp_uib.append(pc_convert(np.sum(bp_data.iloc[-1][0:4]), np.sum(upper_bounds[0][0:4])))
    bp_uib.append(pc_convert(np.sum(upper_bounds[0][0:4]), np.sum(upper_bounds[1][0:4])))
    bp_uib.append(pc_convert(np.sum(upper_bounds[1][0:4]), np.sum(upper_bounds[2][0:4])))
    bp_uib.append(pc_convert(np.sum(upper_bounds[2][0:4]), np.sum(upper_bounds[3][0:4])))
    bp_uib.append(pc_convert(np.sum(upper_bounds[3][0:4]), np.sum(upper_bounds[4][0:4])))
    bp_uib.append(pc_convert(np.sum(upper_bounds[4][0:4]), np.sum(upper_bounds[5][0:4])))

    bp_pf = np.array(bp_pf)
    bp_lib = np.array(bp_lib)
    bp_uib = np.array(bp_uib)

    bp_adjustment = np.array(bp_uib - bp_lib)*bp_correction - (bp_uib - bp_lib)
    bp_adjustment = bp_adjustment / 2
    bp_lib = bp_lib - bp_adjustment
    bp_uib = bp_uib + bp_adjustment

    return bp_pf, bp_lib, bp_uib


def specifier(df1,df2,col1=str,col2=str):
    '''
    Conducts grid test of combinations of two specified columns of forecasted results
    to identify combination that minimizes mean absolute percentage error 
    from actual historic values.

    Parameters
    ----------
    df1, df2 = dataframe outputs from any of the above get_calibration_data functions
        col1, col2 = columns of interest containing forecasted data, should be from
        forecasts of same period out. 
    Returns
    -------
    out : Minimum MAPE value attained, and weights for models one and two used to
        attain minimum. 
    '''
    spec_test = [i/100 for i in range(101)]
    df_local = df1
    mape_list = []
    for i in range(len(spec_test)):
        df_local['loop'] = df1[col1]*spec_test[i]+df2[col2]*(1-spec_test[i]) 
        mape_list.append(mape_calc(df_local,'actual','loop'))
    best_spec = min(mape_list)
    best_index = mape_list.index(best_spec)
    return best_spec, spec_test[best_index], (1-spec_test[best_index])

def get_specifications(m1_data, m2_data, report = True):
    '''
    Runs specifier on each time period generated by forecasts
    to ascertain optimal ensamble weights of each model for each period 
    out from the present
    
    Parameters
    ----------
    m1_data: dataframe containing backtested results from first model
    m2_data: dataframe containing backtested results from second model
    report: if True(default) specifications are printed in addition to returned
    
    Returns
    -------
    out : Numpy arrays containing the optimal weights for each model. 
    '''
    spec_list_a = []
    spec_list_a.append(specifier(m1_data,m2_data,'p1p','p1p')[1])
    spec_list_a.append(specifier(m1_data,m2_data,'p2p','p2p')[1])
    spec_list_a.append(specifier(m1_data,m2_data,'p3p','p3p')[1])
    spec_list_a.append(specifier(m1_data,m2_data,'p4p','p4p')[1])
    spec_list_a.append(specifier(m1_data,m2_data,'p5p','p5p')[1])
    spec_list_a.append(specifier(m1_data,m2_data,'p6p','p6p')[1])

    spec_list_b = []
    spec_list_b.append(specifier(m1_data,m2_data,'p1p','p1p')[2])
    spec_list_b.append(specifier(m1_data,m2_data,'p2p','p2p')[2])
    spec_list_b.append(specifier(m1_data,m2_data,'p3p','p3p')[2])
    spec_list_b.append(specifier(m1_data,m2_data,'p4p','p4p')[2])
    spec_list_b.append(specifier(m1_data,m2_data,'p5p','p5p')[2])
    spec_list_b.append(specifier(m1_data,m2_data,'p6p','p6p')[2])
    if report == True:
        for i in range(1,7):
            print('Best model weighting for period',i,f"is  {spec_list_a[i-1]} for model one and {spec_list_b[i-1]} for model 2.")
    return np.array(spec_list_a), np.array(spec_list_b)

def ensamble_forecast(df, col = str, actual = False):
    '''
    Generates ensamble forecast as final product. Intervals have been calibrated 
    and weighting is done according to grid search for optimal combination.
    default is to report forecast as percentage change from previous format.

    Parameters
    ----------
    df = dataframe with relevant data 
    col = string, name of column in df to forecast
    actual = default to False, if set to True will report actual values of US GDP
    Returns
    -------
    out : point forecast, lower forecast range, upper forecast range
        format = numpy arrays
    '''
    def converter(root, percentage):
        return root * ((percentage / 100) + 1)
    
    as_a_whole = gen_forecast_w(df)
    by_parts = gen_forecast_bp(df)
    specs = get_specifications(get_calibration_data(df, 100), get_by_parts_calibration_data(df, 100), report = False)
    w_weights = specs[0]
    bp_weights = specs[1]

    ensamble_point_forecast = as_a_whole[0] * w_weights + by_parts[0] * bp_weights
    ensamble_lower_range = as_a_whole[1] * w_weights + by_parts[1] * bp_weights
    ensamble_upper_range = as_a_whole[2] * w_weights + by_parts[2] * bp_weights

    if actual == True:
        root = df[col][df.shape[0]-1]
        ensamble_point_forecast[0] = converter(root,ensamble_point_forecast[0])
        for i in range(1,6):
            ensamble_point_forecast[i] = converter(ensamble_point_forecast[i-1],ensamble_point_forecast[i])
        
        ensamble_lower_range[0] = converter(root,ensamble_lower_range[0])
        for i in range(1,6):
            ensamble_lower_range[i] = converter(ensamble_lower_range[i-1],ensamble_lower_range[i])
        
        ensamble_upper_range[0] = converter(root,ensamble_upper_range[0])
        for i in range(1,6):
            ensamble_upper_range[i] = converter(ensamble_upper_range[i-1],ensamble_upper_range[i])
    
    return ensamble_point_forecast, ensamble_lower_range, ensamble_upper_range

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Graphing functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

def add_periods(df, n, delta = 'months'):
    '''
    Helper function to make_ready. Append new dates to end of datetime series with appropriate time interval 
    in between, specified as delta, months by default. 

    Parameters
    ----------
    df: dataframe containing variable to forecast with index set to datetime variable
    
    periods: number of periods out to extend dataset to accomodate forecast

    delta: frequency of time variable, string format, defaults to months

    Returns
    -------


    '''
    updates = [df.index[-1] + relativedelta(months=1)]
    for _ in range(n-1):
        updates.append(updates[-1] + relativedelta(months=1))
    return updates

def make_ready(df,periods):
    '''
    Helper function to graph_forecast. Takes dataframe with datetime index 
    and returns dataframe with original data and extended index in which 
    to append results of forecast

    Parameters
    ----------
    df: dataframe containing variable to forecast with index set to datetime variable
    
    periods: number of periods out to extend dataset to accomodate forecast

    Returns
    -------
    out : dataframe containing original data and time series index extended to 
    encapsulate time window of forecast for use as graphing axis

    '''
    new_dates = add_periods(df,periods)
    dfi = pd.DataFrame(data = None, index = new_dates)
    dfnew = pd.DataFrame(data =None, index = df.index.append(dfi.index))

    fr = pd.merge(dfnew,df, left_index=True, right_index= True, how = 'left')
    return fr

def graph_forecast(df, series, low, high, point, title = str, y_ax = str, p = 6):
    '''
    Graphs results of final forecast output with window of previous data for reference

    Parameters
    ----------
    df = dataframe containing historic data on variable of interest
    series = column within df containing forecasted variable
    low = lower bounds of interval forecast, array or list-like
    high = upper bounds of interval forecast, array or list-like
    point = point forecast, array or list-like
    title = title for graph, string format
    y_ax = label for y axis, string format
    p = number of periods into the future to forecast

    Returns
    -------
    printout of graph

    '''
    
    dfg = pd.DataFrame(df,copy=True)
    g_point = np.insert(point, 0, dfg[series][-1])
    g_low = np.insert(low, 0, dfg[series][-1])
    g_high = np.insert(high, 0, dfg[series][-1])
    
    dfg = make_ready(dfg,p)
    
    fig, ax = plt.subplots(figsize=(12,6))
#     fig.patch.set_facecolor('white')
#     plt.rcParams['figure.facecolor'] = 'white'
    ax.patch.set_facecolor('magenta')
    ax.plot(dfg.index[-(p+1):], g_point, '.-', color = 'cyan', label='Forecast')
    ax.plot(dfg.index[-(p+1):], g_low, '.--', color = 'c', label='Low')
    ax.plot(dfg.index[-(p+1):], g_high, '.-.', color = 'c', label='High')
    ax.fill_between(dfg.index[-(p+1):], g_low, g_high, alpha = .4, color = 'c') # shading interval
    ax.grid(axis='y')
    # ax.plot( sdf.date2[-len(y_test):], y_train, label='actual')
    ax.plot(dfg.index[-50:-p], dfg[series][-50:-p], color='cyan', label='Historic')
    ax.set_xlabel('Time in Months',fontsize = 18)
    ax.set_ylabel(y_ax, fontsize = 18)
    ax.set_title(title, fontsize = 22, pad = 8)
#     ax.legend()
    ax.legend(shadow=1, fontsize='large',loc=2, facecolor = 'm')
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.rcParams['lines.linewidth'] = 2
#     plt.rcParams["font.family"] = "Palatino"
#     plt.rcParams['axes.facecolor'] = 'black'