from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
import numpy as np
import pandas as pd
from dags.preprocessing.SlidingWIndowCV import SlidingWindowSplitCV
from dags.preprocessing.Dataset import Dataset
from catboost import CatBoostRegressor, Pool
import gc


def train(train_data: Dataset, cv: SlidingWindowSplitCV,):

    models = []
    X_train = train_data.get_data()
    y_train = train_data.get_labels()
    # X_test = test_data.get_data()

    mse_result = []
    mae_result = []

    train_predict = 0
    test_predict = []
    feature_importances = []

    for train_idx, valid_idx in cv.split(X_train, 'date_block_num'):
        curr_X_train = X_train.loc[train_idx]
        curr_y_train = y_train.loc[train_idx]

        curr_X_valid = X_train.loc[valid_idx]
        curr_y_valid = y_train.loc[valid_idx]

        train_dataset = Pool(
            curr_X_train,
            curr_y_train,
            cat_features=['date_block_num', 'item_id', 'shop_id',
                         'shop_city', 'shop_type', 'item_category', 'item_subcategory', 'item_fixed_category'],
            text_features=['shop_name', 'item_name', 'item_category_name']
        )

        valid_dataset = Pool(
            curr_X_valid,
            curr_y_valid,
            cat_features=['date_block_num', 'item_id', 'shop_id',
                         'shop_city', 'shop_type', 'item_category', 'item_subcategory', 'item_fixed_category'],
            text_features=['shop_name', 'item_name', 'item_category_name']
        )


        param = {
            # 'iterations': 100,
            # 'depth': 3,
            # 'loss_function': 'RMSE',
            # 'learning_rate': 0.5,
            # 'verbose': True,
            # 'early_stopping_rounds': 5,
            # 'task_type': 'CPU'
            }


        curr_model = CatBoostRegressor(
            early_stopping_rounds=5,
            depth=5,
            learning_rate=0.5,
            task_type='GPU' #TODO change to train locally
        )
        curr_model.fit(train_dataset, eval_set=valid_dataset)

        models.append(curr_model)

        # train_predict = curr_model.predict(curr_X_train)
        # # test_predict.append(curr_model.predict(X_test))

        # mse = mean_squared_error(y_train, train_predict)
        # mae = mean_absolute_error(y_train, train_predict)


        # mse_result.append(mse)
        # mae_result.append(mae)
        # feature_importances.append(curr_model.get_feature_importance())

        # del curr_X_train, curr_y_train, curr_X_valid, curr_y_valid
        # gc.collect()

    return {'models': models, 'test_predict': test_predict, 'mse': mse_result, 'mae': mae_result}