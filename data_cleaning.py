# Author: Doug Hart
# Title: Cleaing data
# Project: Economic Forecasting
# Date Created: 1/3/2021
# Last Updated: 1/9/2021

import pandas as pd
import datetime



def pc_transform(df, srs=str,new=str):
    '''
    Creates new series in dataframe which takes values of percentage change from previous period in
    specified column, intended for time series data. If column contains zeros, issues will ensue resulting in either
    nan or inf values in new column.
    NOTE: series and new MUST be input as strings or errors will ensue, not seemingly an issue with df
    df: name of dataframe in question
    srs: string, name of column from which to measure percentage change
    new: string, name of new series containing transformed data
    
    '''
    df[new] = float(0)
    for i in range(1,df.shape[0]):
        df[new][i] = ((df[srs][i] - df[srs][i - 1]) / df[srs][i - 1])*100
    return df

df = pd.read_csv('use_data.csv')

# creation of trade variables
df['tot'] =  df.exports / df.imports
df['net_exports'] = df.exports - df.imports
# creation of percentage change variables
pc_transform(df,'gdp','pcgdp')
pc_transform(df, 'tot', 'pctot')
pc_transform(df, 'businv', 'pcbusinv')
pc_transform(df, 'C', 'pcC')
pc_transform(df, 'I', 'pcI')
pc_transform(df, 'ip_index', 'pcipi')
# s&p500 variable is already in percentage change form

'''~~~~~~~~~~~~~~~Creating dt formatted date column~~~~~~~~~~~~~~~'''

df['date'] = None
for i in range(df.shape[0]):
    df.date[i] = datetime.date(df.year[i],df.month[i],1)

# To set this var to the df index
df.index = pd.DatetimeIndex(df.date)
df.drop('date',axis=1, inplace = True)

# Useful for viewing all columns in notebook
pd.set_option('display.max_columns', 40)