

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import pandas_profiling
import numpy as np



def check_zero_nan(col, replace = "mean"):
    """
    This function ONLY checks for zero and null values for numerical columns
    Takes col = series or a column from a df, replace = either "mean" or "mode" to be used to replace zero and null values
  
    """
    
    if (col.isna().sum() == 0) & (col[col == 0].count() == 0):
        pass
    else:
        print("column name = ", col.name)
        print("null values = ", col.isna().sum())
        print("zeroes = ", col[col == 0].count())
        display(col.value_counts().sort_values(ascending = False).head(10))
        test = col.copy()
        pre = test.mean()
        print("mean (pre) = " , pre)
        if replace == "mean":
            test.fillna(value = pre, inplace = True)
            test.map(lambda x: pre if x == 0 else x)
            post = test.mean()
            print("after replacing nan and 0 using the average")
            print("mean (post) = ", test.mean())
            print("mean diff = ", round(pre-post, 3), "\n")
        elif replace == "mode": 
            test.fillna(value = test.mode(), inplace = True)
            test.map(lambda x: test.mode() if x == 0 else x)
            post = test.mean()
            print("after replacing nan and 0 using the mode")
            print("mean (post) = ", test.mean())
            print("mean diff = ", round(pre-post, 3), "\n")
        
        

def check_zero_nan_object(col):
    """
    This function ONLY checks for zero and null values for categorical columns; col = series or a column from a df
    """
    
    if (col.isna().sum() == 0) & (col[col == 0].count() == 0):
        pass
    else:
        print("column name = ", col.name)
        print("null values = ", col.isna().sum())
        print("zeroes = ", col[col == 0].count())
        display(col.value_counts().sort_values(ascending = False).head(10))
  
            
     
def binary_col(df, columns):
    """
    This function updates value of columns with 't' and 'f' as unique values; df = dataframe, columns = list of column names to be updated
    """
    for col in columns:
        df[col] = df[col].map(lambda x: 1 if x == 't' else 0)
        df[col] = df[col].astype('int64')
        
        
        
def replace_zero_nan(col, replace):
    """
    This function replaces zero and null values; col = series or a column from a df, replace = either str or float or int to replace zero and null values
    """
    col = col.map(lambda x: replace if (x == 0) else x)
    col.fillna(value = replace, inplace = True)
    return col



def update_money_columns(df, columns):
    """
    This function converts money columns to 'float64'; df = dataframe, columns = list of column names to be updated
    """
    for col in columns:
        df[col] = df[col].map(lambda x: str(x).replace("$", "").replace(",", ""))
        df[col] = df[col].astype("float64")
        
        
def corr_table(df, x, n = 15):
    """
    This function returns a list of top n variables correlated with column 'x'; df = dataframe, x = column in df, n = no of top variables
    """
    corr_values = pd.DataFrame(df.corr()[x])
    corr_values["absolute"] = abs(corr_values[x])
    print("Top ", n, "variables correlated with ", x, ":")
    return pd.DataFrame(corr_values.sort_values(by = "absolute", ascending = False)[x][1:].head(n))