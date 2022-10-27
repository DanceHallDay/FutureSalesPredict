import numpy as np
import pandas as pd
from IDataset import IDataset
import re
from memory_reducer import reduce_memory_usage

class Dataset(IDataset):
    def __init__(self, csv_path : str, train : bool = False) -> None:
        self.train = train
        self.df = ...
        super().__init__()

    def get_data(self):
        pass

    def get_labels(self):
        return super().get_labels()

    def __name_preprocessing__(self, name):
        name = name.lower()
        #delete addition information(name of street)
        name = name.partition('(')[0]
        name = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', name)
        name = name.replace('  ', ' ')
        name = name.strip()
        
        return name    

    def __cat_features_preprocessing__(self, df):

        df['shop_name'] = df['shop_name'].apply(name_preprocessing)
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
        
        return df


    def add_lag_features(train, test, lags, lag_features):
        df = pd.concat([train, test])
        
        lag_features = ['item_revenue', 'item_price', 'item_cnt_month']
        for lag in lags:
                df_lag = df[lag_features + ['date_block_num', 'item_id', 'shop_id']].copy()
                df_lag = df_lag.rename(
                    columns={
                        feature : feature + f'_lag_{lag}'
                        for feature in lag_features
                    }
                )
                
                df_lag['date_block_num'] += lag

                df = pd.merge(
                    df, 
                    df_lag,
                    on = ['item_id', 'shop_id', 'date_block_num'],
                    how = 'left'
                )
        
        reduce_memory_usage(df)
        
        return df


    def drop_outliers(df, cat):
        cat_df = df[df['item_category_id'] == cat]
        cat_df_mean, cat_df_std = cat_df.mean(), cat_df.std()
        cat_df_norm = (cat_df - cat_df_mean) / cat_df_std
        
        cat_df = cat_df[np.abs(cat_df['item_price']) > 3]
        
        return cat_df

    def add_item_cartesian_product():
        pass
        
        
    def etl(files_path, add_item_cartesian_product=False, *args, **kwargs):
        train_dataset = pd.read_csv(files_path + 'sales_train.csv')
        train_dataset['item_revenue'] = train_dataset['item_price'] * train_dataset['item_cnt_day']
        shops = pd.read_csv(files_path + 'shops.csv')
        items = pd.read_csv(files_path + 'items.csv')
        item_cat = pd.read_csv(files_path + 'item_categories.csv')
        
        
        if add_item_cartesian_product:
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

            train_dataset = (
                train_dataset.groupby(['shop_id', 'item_id', 'date_block_num'])
                .agg(
                    {'item_cnt_day' : 'sum', 'item_price' : 'sum', 'item_revenue' : 'sum'}
                )
                .reset_index()
                .rename(columns = {'item_cnt_day' : 'item_cnt_month'})
            )

            train_dataset = pd.merge(grid,train_dataset,how='left',on=['shop_id', 'item_id', 'date_block_num']).fillna(0)
        
        reduce_memory_usage(train_dataset)
        
        train_dataset = pd.merge(train_dataset, items, on="item_id", how="inner")
        train_dataset = pd.merge(train_dataset, shops, on="shop_id", how="inner")
        train_dataset = pd.merge(train_dataset, item_cat, on="item_category_id", how="inner")
        
        train_dataset.drop_duplicates()
        #train_dataset.drop(['date'], axis=1, inplace=True)
        
        #delete neg values in item_price feature
        train_dataset = train_dataset[train_dataset['item_price'] >= 0]
        #train_dataset.loc[train_dataset['item_price'] > 0, 'item_price'].apply(np.log)
        reduce_memory_usage(train_dataset)
        
        #drop outlier
        #for cat in tqdm_notebook(train_dataset['item_category_id'].unique()):
            #train_dataset[train_dataset['item_category_id'] == cat] = drop_outliers(train_dataset, cat)
        
        return train_dataset