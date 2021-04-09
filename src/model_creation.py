# Author: Doug Hart
# Title: VAR creation
# Project: Economic Forecasting
# Date Created: 1/9/2021
# Last Updated: 4/9/2021

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.datetools import dates_from_str

from functions import (get_calibration_data, get_by_parts_calibration_data, 
calibration_check, mape_calc, report_calibration, volatility, specifier, 
gen_forecast_w, gen_forecast_bp, get_specifications, ensamble_forecast)

# Useful for viewing all columns in notebook
# pd.set_option('display.max_columns', 40)


df = pd.read_pickle('cleaned_421.pkl', compression='zip')


'''~~~~~~~~~~~~~~~~~~~~~~~~~~Part 1: Basic VAR Model Creation Code~~~~~~~~~~~~~~~~~~~~~~~~~~'''
# Target dataset creation and time formatting
dfc = pd.DataFrame(df,copy=True)
mdata = dfc[['pcgdp','mancap','unem','pctot','pcbusinv','pcC','pcI','pcipi','pcsp500']]
dates = dfc[['year', 'month']].astype(int).astype(str)
dates.reset_index(inplace=True,drop=True)
monthly = dates['year'] + "M" + dates['month']
monthly = dates_from_str(monthly)
mdata.index = pd.DatetimeIndex(monthly)
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

'''~~~~~~~~~~~~~~~~~~~~~Part 2: Generating Out-of-Sample Forecasts for Model Evaluation~~~~~~~~~~~~~~~~~~~~~'''



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~By Parts~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''


# V2 actually seems to make worse interval predictions in first period, but more accurate
# predictins in subsequent periods. This result holds whether results fitted to six lags or 
# IC determined fit. Next step to check MAPE scores 
# to eval accuracy of point predictions. 




for i in ['C','I','G','net_exports','unem','meanprice','mancap','man_industelect','electtot']:
    volatility(i, report=True)
# unem is the only standout for high volatility in this list
# others in V2 list of perc change versions




'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Part 3: Generating Calibrated Forecasts~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''




'''
Best model weighting for period 1 is  1.0 for model one and 0.0 for model two.
Best model weighting for period 2 is  0.97 for model one and 0.030000000000000027 for model two.
Best model weighting for period 3 is  1.0 for model one and 0.0 for model two.
Best model weighting for period 4 is  0.62 for model one and 0.38 for model two.
Best model weighting for period 5 is  0.31 for model one and 0.69 for model two.
Best model weighting for period 6 is  0.73 for model one and 0.27 for model two.

In this case model one is as a whole and model two is by parts
'''

