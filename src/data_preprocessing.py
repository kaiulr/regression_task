import pandas as pd
import numpy as np
import os
from pathlib import Path

def preprocessing(input_dir, output_dir):
    df = pd.read_csv(input_dir)

    ## Checking missing values and dtypes
    null_cells = df.isnull().sum()
    print("Number of Null Values in Each Column:")
    print(null_cells)
    
    # No null cells found in the data set provided. If the test has null values, we can interpolate by running this command:
    if null_cells.sum() != 0:
        df.fillna(df.mean(), inplace=True)

    dtypes = df.dtypes
    print(dtypes)
    print("Column Datatypes:")
    # All dtypes in columns appear correct (training data)

    # Noticed 'COEMISSIONS ' column has extra whitespace
    df = df.rename(columns={'COEMISSIONS ':'COEMISSIONS'})

    ## Creating dummy variables for categorical variables, with <10 categories, for analysis.
    category_10_columns = [column for column in df.columns if (df[column].dtype=='object' and df[column].nunique()<=10)]
    df = pd.get_dummies(df, columns=category_10_columns, drop_first=True)

    ## Dropping year as it adds no information
    df = df.drop(columns=['Year'])

    # Min-max scaling continuous variables (in order to help prediction accuracy and understanding of Betas)
    df['scaled_fuel_consumption'] = (df['FUEL CONSUMPTION'] - min(df['FUEL CONSUMPTION']))/(max(df['FUEL CONSUMPTION'])-min(df['FUEL CONSUMPTION']))
    df['scaled_engine_size'] = (df['ENGINE SIZE'] - min(df['ENGINE SIZE']))/(max(df['ENGINE SIZE'])-min(df['ENGINE SIZE']))
    df['scaled_coemissions'] = (df['COEMISSIONS'] - min(df['COEMISSIONS']))/(max(df['COEMISSIONS'])-min(df['COEMISSIONS']))

    df.to_csv(os.path.join(output_dir, 'training_data.csv'), index=False)
    print(f"Processed data saved to '{os.path.join(output_dir, 'training_data.csv')}'")

    return os.path.join(output_dir, 'training_data.csv')