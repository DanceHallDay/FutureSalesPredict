from sklearn.metrics import (
    # mean_absolute_error,
    mean_squared_error
)
import numpy as np
import pandas as pd
from dags.preprocessing.SlidingWIndowCV import SlidingWindowSplitCV
from dags.preprocessing.Dataset import Dataset
from catboost import CatBoostRegressor, Pool
import neptune.new as neptune
import gc



run = neptune.init_run(
    project="lev.taborov/FutureSalesPredict",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNTQ5YzlhNS03NDNiLTRjYmItYmQ5Ni1lMWViOTViNjllNmIifQ==",
)  # your credentials




def train(train_data: Dataset, cv: SlidingWindowSplitCV,):


    X_train = train_data.get_data().drop(columns='item_cnt_day')
    y_train = train_data.get_labels()

    run['train/used_columns'].log(
        X_train.columns
    )

    train_predict = 0


    param = {
            'iterations': 1000,
            'depth': 10,
            'loss_function': 'RMSE',
            'learning_rate': 0.07,
            'verbose': True,
            'early_stopping_rounds': 10,
            'task_type': 'GPU'
            }

    run['parameters'].log(str(param))


    model = CatBoostRegressor(**param)

    for train_idx, valid_idx in cv.split(X_train, 'date_block_num'):
        curr_X_train = X_train.loc[train_idx]
        curr_y_train = y_train.loc[train_idx]

        curr_X_valid = X_train.loc[valid_idx]
        curr_y_valid = y_train.loc[valid_idx]

        train_dataset = Pool(
            curr_X_train,
            curr_y_train,
            cat_features=[
                'item_id',
                'shop_id',
                'shop_city',
                'shop_type',
                'item_category',
                'item_subcategory',
                'item_fixed_category'
            ],
            text_features=[
                # 'shop_name',
                # 'item_name',
                # 'item_category_name'
            ]
        )

        valid_dataset = Pool(
            curr_X_valid,
            curr_y_valid,
            cat_features=[
                'item_id',
                'shop_id',
                'shop_city',
                'shop_type',
                'item_category',
                'item_subcategory',
                'item_fixed_category'
            ],
            text_features=[
                # 'shop_name',
                # 'item_name',
                # 'item_category_name'
            ]
        )




        model.fit(
            train_dataset,
            eval_set=valid_dataset
        )


        train_predict = model.predict(curr_X_valid)


        mse = mean_squared_error(curr_y_valid, train_predict)
        # mae = mean_absolute_error(y_train, train_predict)
        run['train/mse'].log(mse)


        # run['feature_importance'].log(
        #     model.get_feature_importance()
        # )

        del curr_X_train, curr_y_train, curr_X_valid, curr_y_valid
        gc.collect()

    # return {'models': models[-1], 'test_predict': test_predict, 'mse': mse_result, 'mae': mae_result}

    return model

