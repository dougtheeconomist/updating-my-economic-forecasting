# Author: Doug Hart
# Title: Cleaing data
# Project: Economic Forecasting
# Date Created: 1/3/2021
# Last Updated: 4/9/2021

import pandas as pd
import datetime

df = pd.read_csv('use_data.csv')

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
