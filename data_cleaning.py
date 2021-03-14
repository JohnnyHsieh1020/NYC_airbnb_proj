# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 21:18:57 2021

@author: Johnny Hsieh
"""

import pandas as pd

df = pd.read_csv('AB_NYC_2019.csv')
df.columns
# Drop irrelevant columns
df = df.drop(columns=['id', 'host_name', 'last_review'], axis = 1)

# Check null
df.isnull().sum()

# Use mean to replace the missing data ('reviews_per_month' column)
mean = df['reviews_per_month'].mean()
df['reviews_per_month'].fillna(mean, inplace=True)

# Drop missing data ('name' column)
df = df.dropna(axis=0)

# Calculate 'name' length
df['name_len'] = df['name'].apply(lambda x: len(x))

# Export to CSV
df.to_csv('AB_NYC_cleaned.csv', index = False)