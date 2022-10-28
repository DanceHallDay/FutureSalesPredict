import numpy as np
import pandas as pd

def reduce_memory_usage(df:pd.DataFrame, verbose:int=True):
    numeric_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    curr_memory_usage = df.memory_usage().sum() / 1024**2 
    
    for column in df.columns:
        column_type = df[column].dtypes.name
        if column_type in numeric_types:
            column_min = df[column].min()
            column_max = df[column].max()
            
            if 'int' in column_type:
                if column_min > np.iinfo(np.int8).min and column_max < np.iinfo(np.int8).max:
                    df[column] = df[column].astype(np.int8)
                    
                elif column_min > np.iinfo(np.int16).min and column_max < np.iinfo(np.int16).max:
                    df[column] = df[column].astype(np.int16)
                    
                elif column_min > np.iinfo(np.int32).min and column_max < np.iinfo(np.int32).max:
                    df[column] = df[column].astype(np.int32)
                    
                elif column_min > np.iinfo(np.int64).min and column_max < np.iinfo(np.int64).max:
                    df[column] = df[column].astype(np.int64)
                    
            else:
                if column_min > np.iinfo(np.int8).min and column_max < np.iinfo(np.int8).max:
                    df[column] = df[column].astype(np.float16)
                    
                elif column_min > np.iinfo(np.int16).min and column_max < np.iinfo(np.int16).max:
                    df[column] = df[column].astype(np.float16)
                    
                elif column_min > np.iinfo(np.int32).min and column_max < np.iinfo(np.int32).max:
                    df[column] = df[column].astype(np.float32)
                    
                elif column_min > np.iinfo(np.int64).min and column_max < np.iinfo(np.int64).max:
                    df[column] = df[column].astype(np.float64)
                    
    final_memory_usage = df.memory_usage().sum() / 1024**2 
            
    if verbose:
        print(f'Memory usage dicreased from {curr_memory_usage:.2f} Mb to {final_memory_usage:.2f} Mb.')
            
    return df 