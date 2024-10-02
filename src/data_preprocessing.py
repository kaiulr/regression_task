import pandas as pd
import numpy as np
import os
from pathlib import Path


output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'data') 

df = pd.read_csv("C:/Users/Fiona/Desktop/Fiona_Arora_A1/fuel_train.csv")

## Checking missing values and dtypes
null_cells = df.isnull().sum()
print(null_cells)
# No null cells found

dtypes = df.dtypes
print(dtypes)
# All dtypes in columns appear correct.

# Noticed 'COEMISSIONS ' column has extra whitespace
df = df.rename(columns={'COEMISSIONS ':'COEMISSIONS'})

## Creating dummy variables for categorical variables, with <10 categories, for analysis.
category_10_columns = [column for column in df.columns if (df[column].dtype=='object' and df[column].nunique()<=10)]
df = pd.get_dummies(df, columns=category_10_columns, drop_first=True)

## Dropping year as it adds no information
df = df.drop(columns=['Year'])

df.to_csv(os.path.join(output_dir, 'training_data.csv'), index=False)

# Normalizing continuous variables