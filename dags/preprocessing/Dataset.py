import numpy as np
import pandas as pd
# from sklearn import preprocessing
import re
from .memory_reducer import reduce_memory_usage
# from itertools import product
from typing import List


class Dataset:
    def __init__(self, file_path: str, train: bool = False, *args, **kwargs) -> None:

        self.train = train
        self.file_path = file_path

        self.__read_data__()
        self.__etl__()



    def __read_data__(self) -> None:
        """Reads data from file
        """
        # harcoded, maybe should be done in a better way
        if self.train:
            self.df = pd.read_csv(self.file_path + '/sales_train.csv')
        else:
            self.df = pd.read_csv(self.file_path + '/test.csv')

        self.items = pd.read_csv(self.file_path + '/items.csv')
        self.item_categories = pd.read_csv(self.file_path + '/item_categories.csv')
        self.shops = pd.read_csv(self.file_path + '/shops.csv')



    def __etl__(self) -> None:

        dataset = self.df

        # fixing data problems from DQC
        # 1. drop outliers
        if self.train:
            self.__drop_outliers__()


        # 2. numerical data transform
        # TODO figure out data transformation

        # 3. categorical data transform
        self.shops['shop_name'] = self.shops['shop_name'].apply(self.__shop_name_preprocessing__)
        self.shops['shop_city'] = self.shops['shop_name'].apply(lambda x: x.split(' ')[0])

        self.__cat_features_preprocessing__()

        self.df = self.df
        self.df = pd.merge(self.df, self.items, on="item_id")
        self.df = pd.merge(self.df, self.shops, on="shop_id", how="left")
        self.df = pd.merge(self.df, self.item_categories, on="item_category_id", how="left")
        self.df.drop_duplicates()

        if not self.train:
            self.df['date_block_num'] = 34

        columns = [
            'shop_id',
            'item_id',
            'date_block_num',
            # 'shop_name',
            'shop_city',
            'shop_type',
            # 'item_name',
            # 'item_price', #TODO think about price
            'item_category_id',
            # 'item_category_name',
            'item_category',
            'item_subcategory',
            'item_fixed_category'
        ]
        # dataset without price
        if self.train:
            columns.append('item_cnt_day')
            self.df = self.df[columns]
            self.df['item_cnt_day'].apply(np.log10)
        else:
            self.df = self.df[columns]

        reduce_memory_usage(self.df)


    def get_data(self) -> pd.DataFrame:
        '''_summary_

        Parameters
        ----------
        add_cat_preprocessing : bool, optional
            _description_, by default False
        add_lag_features : bool, optional
            _description_, by default False

        Returns
        -------
        pd.DataFrame
            _description_
        '''
        return self.df


    def get_labels(self) -> pd.Series:
        '''Returns target labels
        '''
        return self.df['item_cnt_day']

    def __shop_name_preprocessing__(self, name: str) -> str:

        name = name.lower()
        # delete addition information(name of street)
        name = name.partition('(')[0]
        name = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', name)
        name = name.replace('  ', ' ')
        name = name.strip()

        return name

    def __fix_category__(self, category_id: str) -> str:

        if category_id == 0:
            return 'Headphones'
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

    def __drop_outliers__(self) -> None:

        outliers = self.df[
            (self.df['item_price'] < 0) |
            (self.df['item_price'] > 100_000) |
            (self.df['item_cnt_day'] > 1000)]
        self.df.drop(outliers.index)

    def __cat_features_preprocessing__(self) -> None:


        self.shops.loc[self.shops['shop_city'] == 'н', 'shop_city'] = 'нижний новгород'
        self.shops.loc[self.shops['shop_name'].str.contains('новгород'), 'shop_name'] = (
            self.shops.loc[self.shops['shop_name'].str.contains('новгород'), 'shop_name'].
            apply(lambda x: x.replace('новгород', ''))
        )

        self.shops['shop_type'] = self.shops['shop_name'].apply(
            lambda x: x.split()[1] if (len(x.split()) > 1) else 'other'
        )
        self.shops.loc[
            (self.shops['shop_type'] == 'орджоникидзе') |
            (self.shops['shop_type'] == 'ул') |
            (self.shops['shop_type'] == 'распродажа') |
            (self.shops['shop_type'] == 'торговля'),
            'shop_type'
        ] = 'other'

        self.item_categories['item_category'] = (
            self.item_categories['item_category_name']
            .str.split(' - ').apply(lambda x: x[0])
        )
        self.item_categories['item_subcategory'] = (
            self.item_categories['item_category_name']
            .str.split(' - ').apply(lambda x: x[-1])
        )

        self.item_categories['item_fixed_category'] = self.item_categories['item_category_id'].apply(
            self.__fix_category__)



    def __add_lag_features__(self, df: pd.DataFrame, other_df: List[pd.DataFrame], lags: list, lag_features: list) -> pd.DataFrame:
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
                    on=['item_id', 'shop_id', 'date_block_num'],
                    how='left'
                )


        return result_df

