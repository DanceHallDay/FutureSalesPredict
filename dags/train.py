from preprocessing.IDataset import IDataset
from dags.preprocessing.SlidingWIndowCV import SlidingWindowSplitCV
from sklearn.metrics import (
    mean_absolute_error,
     mean_squared_error
)
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc


def train_lgb(train_data:IDataset, test_data:IDataset, cv:SlidingWindowSplitCV,):
    models = []
    X_train = train_data.get_data()
    y_train = train_data.get_labels()
    X_test = test_data.get_data()
    mse_result = []
    mae_result = []
    test_predict = []
    train_predict = 0
    feature_importances = []
    
    for train_idx, valid_idx in cv.split(X_train, 'date_block_num'):
        curr_X_train = X_train.loc[train_idx]
        curr_y_train = y_train.loc[train_idx]
        curr_X_valid = X_train.loc[valid_idx]
        curr_y_valid = y_train.loc[valid_idx]
        
        train_dataset = lgb.Dataset(
            curr_X_train,
            label=curr_y_train,
            feature_name=X_train.columns.tolist(),
            categorical_feature=['date_block_num', 'item_id', 'shop_id','shop_city','shop_type','item_category','item_subcategory'],
            #free_row_data=False
        )
        
        valid_dataset = lgb.Dataset(
            curr_X_valid,
            label=curr_y_valid,
            feature_name=X_train.columns.tolist(),
            categorical_feature=['date_block_num', 'item_id', 'shop_id', 'shop_city','shop_type','item_category','item_subcategory'],
            #free_row_data=False
        )
        
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'force_col_wise': 'true'
        }
        
        lgb_reg = lgb.train(
            params,
            train_set=train_dataset,
            num_boost_round=1000,
            valid_sets=[train_dataset, valid_dataset],
            verbose_eval=25, # verbose_param
            early_stopping_rounds=50
        )
        
        models.append(lgb_reg)

        train_predict = lgb_reg.predict(curr_X_train, num_iteration=lgb_reg.best_iteration) 
        test_predict.append(
            lgb_reg.predict(X_test, num_iteration=lgb_reg.best_iteration)
        )
        
        mse = mean_squared_error(y_train, train_predict)
        mae = mean_absolute_error(y_train, train_predict)
        

        mse_result.append(mse)
        mae_result.append(mae) 
        feature_importances.append(lgb_reg.feature_importance())
        
        del curr_X_train, curr_y_train, curr_X_valid, curr_y_valid
        gc.collect()
        
    return models, test_predict