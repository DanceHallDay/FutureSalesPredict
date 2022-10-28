
import numpy as np
import pandas as pd
from sklearn import preprocessing
from IDataset import IDataset
import re
from memory_reducer import reduce_memory_usage
from itertools import product
from typing import List

class Dataset(IDataset):
    def __init__(self, csv_path:str, description_files_path:str, train:bool=False) -> None:
        self.train = train
        self.df = self.__etl__(csv_path)
        super().__init__()

    def get_data(self, add_cat_preprocessing:bool=True, add_lag_features:bool=False, *args, **kwargs) -> pd.DataFrame:
        result_df =  self.df.drop(['item_cnt_month', 'item_price', 'item_revenue'], axis=1) if self.train else self.df
        result_df = self.__cat_features_preprocessing__(result_df) if add_cat_preprocessing else result_df
        result_df = self.__add_lag_features__(result_df, *args, **kwargs) if add_lag_features else result_df
        return result_df

    def get_labels(self) -> pd.Series:
        return self.df.item_cnt_month if self.train else None

    def __name_preprocessing__(self, name : str) -> str:
        name = name.lower()
        #delete addition information(name of street)
        name = name.partition('(')[0]
        name = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', name)
        name = name.replace('  ', ' ')
        name = name.strip()
        
        return name    

    def __fix_category__(self, category_id:int) -> str:
        if category_id == 0:
            return "Headphones"
        elif category_id in range(1, 8):
            return 'Accessory'
        elif category_id == 8:
            return 'Tickets'
        elif category_id == 9:
            return 'Delivery'
        elif category_id in range(10, 18):
            return 'Consoles'
        elif category_id in range(18, 32):
            return 'Games'
        elif category_id in range(32, 37):
            return 'Pay Card'
        elif category_id in range(37, 42):
            return 'Films'
        elif category_id in range(42, 54):
            return 'Books'
        elif category_id in range(54, 61):
            return 'Music'
        elif category_id in range(61, 73):
            return 'Gifts'
        elif category_id in range(73, 79):
            return 'Soft'
        elif category_id in range(79, 81):
            return 'Music'
        elif category_id in range(81, 83):
            return 'Clean'
        else:
            return 'Charging'
    
    def __cat_features_preprocessing__(self, df:pd.DataFrame) -> pd.DataFrame:

        df['shop_name'] = df['shop_name'].apply(self.__name_preprocessing__)
        df['shop_city'] = df['shop_name'].apply(lambda x : x.split(' ')[0])

        df.loc[df['shop_city'] == 'н', 'shop_city'] = 'нижний новгород'
        df.loc[df['shop_name'].str.contains('новгород'), 'shop_name'] = (
            df.loc[df['shop_name'].str.contains('новгород'), 'shop_name'].
            apply(lambda x : x.replace('новгород', ''))
        )

        df['shop_type'] = df['shop_name'].apply(
            lambda x : x.split()[1] if (len(x.split())>1) else 'other'
        )
        df.loc[
            (df['shop_type'] == 'орджоникидзе') |
            (df['shop_type'] == 'ул') |
            (df['shop_type'] == 'распродажа') |
            (df['shop_type'] == 'торговля'),
            'shop_type'
        ] = 'other'

        df['item_category'] = (
            df['item_category_name']
            .str.split(' - ').apply(lambda x: x[0])
        )
        df['item_subcategory'] = (
            df['item_category_name']
            .str.split(' - ').apply(lambda x: x[-1])
        )

        df['item_fixed_category'] = df['item_category_id'].apply(self.fix_category)
        
        return df


    def __add_lag_features__(df:pd.DataFrame, other_df:List[pd.DataFrame], lags:list, lag_features:list) -> pd.DataFrame:
        result_df = pd.concat([df, *other_df])
        
        lag_features = ['item_revenue', 'item_price', 'item_cnt_month']
        for lag in lags:
                df_lag = result_df[lag_features + ['date_block_num', 'item_id', 'shop_id']].copy()
                df_lag = df_lag.rename(
                    columns={
                        feature : feature + f'_lag_{lag}'
                        for feature in lag_features
                    }
                )
                
                df_lag['date_block_num'] += lag

                result_df = pd.merge(
                    result_df, 
                    df_lag,
                    on = ['item_id', 'shop_id', 'date_block_num'],
                    how = 'left'
                )
        
        reduce_memory_usage(result_df)
        
        return result_df


    def __drop_outliers__(df:pd.DataFrame, cat:str) -> pd.DataFrame:
        cat_df = df[df['item_category_id'] == cat]
        cat_df_mean, cat_df_std = cat_df.mean(), cat_df.std()
        cat_df_norm = (cat_df - cat_df_mean) / cat_df_std
        
        cat_df = cat_df[np.abs(cat_df['item_price']) > 3]
        
        return cat_df
        
    def __etl__(self, file_path : str, add_item_cartesian_product : bool = False) -> pd.DataFrame:
        dataset = pd.read_csv(file_path)
        if self.train:
            dataset['item_revenue'] = train_dataset['item_price'] * train_dataset['item_cnt_day']
        shops = pd.read_csv(self.description_files_path)
        items = pd.read_csv(self.description_files_path)
        item_cat = pd.read_csv(self.description_files_path)
        
        
        if add_item_cartesian_product and self.train:
            grid = []
            for month in range(34):
                unique_shops = train_dataset.loc[
                    train_dataset['date_block_num'] == month,
                    'shop_id'
                ].unique()
                unique_items = train_dataset.loc[
                    train_dataset['date_block_num'] == month,
                    'item_id'
                ].unique()
                
                grid.append( 
                    np.array(
                        list(product(*[unique_shops, unique_items, [month]]))
                    )
                )
                    
            grid = pd.DataFrame(
                np.vstack(grid),
                columns=['shop_id', 'item_id', 'date_block_num'],
                dtype=np.int32
            )

            dataset = (
                dataset.groupby(['shop_id', 'item_id', 'date_block_num'])
                .agg(
                    {'item_cnt_day' : 'sum', 'item_price' : 'sum', 'item_revenue' : 'sum'}
                )
                .reset_index()
                .rename(columns = {'item_cnt_day' : 'item_cnt_month'})
            )

            train_dataset = pd.merge(grid,train_dataset,how='left',on=['shop_id', 'item_id', 'date_block_num']).fillna(0)
        
        reduce_memory_usage(dataset)
        
        dataset = pd.merge(dataset, items, on="item_id", how="inner")
        dataset = pd.merge(dataset, shops, on="shop_id", how="inner")
        dataset = pd.merge(dataset, item_cat, on="item_category_id", how="inner")
        
        dataset.drop_duplicates()
        #train_dataset.drop(['date'], axis=1, inplace=True)
        
        #delete neg values in item_price feature
        dataset = dataset[train_dataset['item_price'] >= 0]
        #train_dataset.loc[train_dataset['item_price'] > 0, 'item_price'].apply(np.log)
        reduce_memory_usage(dataset)
        
        #drop outlier
        #for cat in tqdm_notebook(train_dataset['item_category_id'].unique()):
            #train_dataset[train_dataset['item_category_id'] == cat] = drop_outliers(train_dataset, cat)
        
        return dataset