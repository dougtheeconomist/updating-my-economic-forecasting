# Author: Doug Hart
# Title: Cleaing data
# Project: Economic Forecasting
# Date Created: 1/3/2021
# Last Updated: 4/19/2021

import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import requests
from secrets import FRED_api_key
from secrets import eia_api_key

# Loading the data
df = pd.read_csv('use_data.csv')

# Creating date column
df['date'] = None
for i in range(df.shape[0]):
    df.date[i] = datetime.date(df.year[i],df.month[i],1)

'''~~~~~~~~~~~~~~~~~~~~~~~~~Updating data from relevant APIs~~~~~~~~~~~~~~~~~~~~~~~~~'''
##########
#FRED_api#
##########

# Documentation:
# https://fred.stlouisfed.org/docs/api/fred/series_observations.html

# Below is an example HTTP GET request provided in documentation:
f'https://api.stlouisfed.org/fred/series/search?api_key={FRED_api_key}&search_text=canada'

# to add: observation_start frequency=m aggregation_method=sum or avg
# Testing connection
api_url = f'https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&api_key={FRED_api_key}&file_type=json'
data = requests.get(api_url)
print(data.status_code)

next_date = str(df.date[df.shape[0]-1] + relativedelta(months=1))
FRED_series = ['MCUMFN', 'UNRATE', 'INDPRO','SP500' ,'ASPNHSUS']

fewest_updates = 'N/A' #taking least number of months of new data
new_data = {}
# Looping through because of lack of batch call functionality, shouldn't be issue for 5 series
for series in FRED_series:
    val_list = []
    api_url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series}&observation_start={next_date}&frequency=m&aggregation_method=avg&api_key={FRED_api_key}&file_type=json'
    data = requests.get(api_url).json()
    for i in range(len(data['observations'])):
        val_list.append(data['observations'][i]['value'])
    new_data[series] = val_list
    if fewest_updates == 'N/A':
        fewest_updates = len(data['observations'])
    else: 
        fewest_updates = min(len(data['observations']), fewest_updates)


#########
#EIA_api#
#########
# Documentation:
# https://fred.stlouisfed.org/docs/api/fred/

# Sample url: f'http://api.eia.gov/series/?series_id=sssssss&api_key={eia_api_key}[&num=][&out=json]'
# Multiple series can be fetched in a single request by using a semi-colon separated list of 
# series id's. The number of series in a single request is limited to 100.

eia_series_codes = [
    'STEO.GDPQXUS.M', 
    'STEO.TREXRUS.M', 
    'STEO.KRDRXUS.M', 
    'STEO.CONSRUS.M', 
    'STEO.I87RXUS.M', 
    'STEO.GOVXRUS.M', 
    'STEO.TRIMRUS.M', 
    'STEO.TOTOGEN_US.M', 
    'STEO.ELICP_US.M']

# Checking connection
api_url = f'http://api.eia.gov/series/?api_key={eia_api_key}&series_id=STEO.GDPQXUS.M&out=json'
data = requests.get(api_url)
print(data.status_code)

# Batch url to grab all required data series
api_url = f'http://api.eia.gov/series/?api_key={eia_api_key}&series_id=STEO.GDPQXUS.M;STEO.TREXRUS.M;STEO.KRDRXUS.M;STEO.CONSRUS.M;STEO.I87RXUS.M;STEO.GOVXRUS.M;STEO.TRIMRUS.M;STEO.TOTOGEN_US.M;STEO.ELICP_US.M&out=json'
data = requests.get(api_url).json()
# Parsing through and saving relevant data from each series
eia_y = []
eia_m = []

for chunk in range(len(data['series'])):
    # Parsing window of new data from older data and projected data
    newest_relevant = data['series'][chunk]['lastHistoricalPeriod']
    oldest_relevant = next_date[:4] + next_date[5:7]
    for i in range(len(data['series'][chunk]['data'])):
        if data['series'][chunk]['data'][i][0] == newest_relevant:
            first_cut = i
            print(data['series'][chunk]['data'][i],f'n/r for {data['series'][chunk]['name']}')
        if data['series'][chunk]['data'][i][0] == oldest_relevant:
            second_cut = i + 1
            print(data['series'][chunk]['data'][i], f'o/r for {data['series'][chunk]['name']}')
            break

    relevant_data = data['series'][chunk]['data'][first_cut:second_cut]
    relevant_data.reverse()
    if chunk == 0:
        for i in range(len(relevant_data)):
            eia_y.append(int(relevant_data[i][0][:4]))
            eia_m.append(int(relevant_data[i][0][4:]))
    for i in range(len(relevant_data)):
        relevant_data[i] = relevant_data[i][1]
    relevant_data

    if fewest_updates > len(relevant_data):
        fewest_updates = len(relevant_data)
    
    # Adding entry to dict with parsed series name as the key
    new_data[data['series'][chunk]['name']] = relevant_data
new_data['year'] = eia_y
new_data['month'] = eia_m
'''
Index(['year', 'month', 'gdp', 'C', 'I', 'G', 'imports', 'exports',
       'meanprice', 'unem', 'mancap', 'industelect', 'electtot', 'pcsp500',
       'ip_index', 'businv', 'date'])
FRED_series = ['MCUMFN', 'UNRATE', 'INDPRO','SP500' ,'ASPNHSUS']
'''

# Updating new data to dataframe according to new observation count with least updated series
for i in range(fewest_updates):
    df = df.append(pd.Series([
        new_data['year'][i], 
        new_data['month'][i], 
        new_data['Real Gross Domestic Product, Monthly'][i], 
        new_data['Real Personal Consumption Expenditures, Monthly'][i], 
        new_data['Real Private Fixed Investment, Monthly'][i], 
        new_data['Real U.S. Government Expenditures, Monthly'][i], 
        new_data['Real Imports of Goods and Services, Monthly'][i], 
        new_data['Real Exports of Goods and Services, Monthly'][i], 
        new_data['ASPNHSUS'][i], 
        new_data['UNRATE'][i], 
        new_data['MCUMFN'][i], 
        new_data['Retail Sales of Electricity in Industrial Sector, U.S. Total, Monthly'][i], 
        new_data['Total net generation by all energy sources, all sectors, United States, Monthly'][i], 
        new_data['SP500'][i], 
        new_data['INDPRO'][i], 
        new_data['Business Inventory Change, Monthly'][i], 
        datetime.date(df.year[i],df.month[i],1)

    ], index = df.columns), ignore_index = True)

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Feature Engineering~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
# Useful for viewing all columns in notebook
# pd.set_option('display.max_columns', 40)

# creation of trade variables
df['tot'] =  df.exports / df.imports
df['net_exports'] = df.exports - df.imports

df['man_industelect'] = df.mancap*df.industelect
# creation of percentage change variables
transformations = {
    'gdp':'pcgdp', 
    'C':'pcC', 
    'I':'pcI', 
    'G':'pcG', 
    'net_exports':'pc_ne', 
    'tot':'pctot', 
    'businv':'pcbusinv', 
    'meanprice':'pc_mp', 
    'mancap':'pc_mc', 
    'man_industelect':'pc_mie', 
    'electtot':'pc_et', 
    'ip_index':'pcipi'}


for i in transformations.keys():
    df[transformations[i]] = df[i].pct_change()*100
    df[transformations[i]][0] = 0

# s&p500 variable is already in percentage change form
'''~~~~~~~~~~~~~~~Setting date column as index~~~~~~~~~~~~~~~'''

# To set date var to the df index
df.index = pd.DatetimeIndex(df.date)
df.drop('date',axis=1, inplace = True)