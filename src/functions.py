# Author: Doug Hart
# Title: VAR functions
# Project: Economic Forecasting
# Date Created: 4/8/2021
# Last Updated: 4/8/2021

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base.datetools import dates_from_str
