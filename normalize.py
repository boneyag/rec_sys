import pandas as pd

def normalize_col(df_list, column_name):

    for df in df_list:
        
        df[column_name]=(df[column_name]-df[column_name].min())/(df[column_name].max()-df[column_name].min())
        
    return df_list