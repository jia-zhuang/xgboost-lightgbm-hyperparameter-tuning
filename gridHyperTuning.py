import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
import xgboost as xgb
import fire
from sklearn.model_selection import ParameterGrid
import pickle

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    else:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def optimize_xgb(data, target, log_pickle):
    trn_data = xgb.DMatrix(data, label=target)

    param_range = {
        'max_depth': range(4, 10),
        'min_child_weight': range(10, 150, 30),
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }

    start_time = timer()
    
    log = []
    for param in ParameterGrid(param_range):
        param.update({
            'objective': 'reg:linear',
            'booster': 'gbtree',
            'eval_metric': 'rmse',
            'learning_rate': 0.01,
            'silent': 1, 
        })
        xgb_cv = xgb.cv(param, trn_data, 10000, nfold=4, metrics=('rmse'), early_stopping_rounds=600, verbose_eval=200)
        
        log.append((xgb_cv['test-rmse-mean'].idxmin(), xgb_cv['test-rmse-mean'].min(), param))
        print(f'\nParamGrid:\t{log[-1][0]}\t{log[-1][1]}\t{param["max_depth"]}\t{param["min_child_weight"]}\t{param["subsample"]}\t{param["colsample_bytree"]}\n')

    with open(log_pickle, 'wb') as f:
        pickle.dump(log, f)

    timer(start_time)


def optimize_lgb(data, target, log_pickle, feature_name='auto', categorical_feature='auto'):
    parm_range = {
        'num_leaves': [30, 50, 100, 150],
        'min_data_in_leaf': [30, 50, 100, 150],
        'bagging_fraction': [0.7, 0.8, 0.9],
        'feature_fraction': [0.7, 0.8, 0.9],
        'lambda_l1': [0, 1, 10]
    }
    start_time = timer()
    log = []
    for param in ParameterGrid(parm_range):
        param.update({
            'objective': 'regression',
            'boosting': 'gbdt',
            'metric': 'rmse',
            'learning_rate': 0.01,
        })
        trn_data = lgb.Dataset(data, label=target, feature_name=feature_name, categorical_feature=categorical_feature)
        lgb_cv = lgb.cv(param, trn_data, 10000, nfold=4, stratified=False, early_stopping_rounds=600, verbose_eval=200)
        log.append((np.argmin(lgb_cv['rmse-mean']), np.min(lgb_cv['rmse-mean']), param))
        print(f'\nParamGrid:\t{log[-1][0]}\t{log[-1][1]}\t{param["num_leaves"]}\t{param["min_data_in_leaf"]}\t{param["bagging_fraction"]}\t{param["feature_fraction"]}\t{param["lambda_l1"]}\n')

    with open(log_pickle, 'wb') as f:
        pickle.dump(log, f)
    
    timer(start_time)

def main(train_csv, target_txt, log_pickle):
    train = pd.read_csv(train_csv, index_col=0)
    target = np.loadtxt(target_txt)
    feats = train.columns[2:].tolist()
    cat_feats = ['feature_1', 'feature_2', 'feature_3']
    optimize_lgb(train.iloc[:, 2:].values, target, log_pickle, feats, cat_feats)
    #optimize_xgb(train.iloc[:, 2:].values, target, log_pickle)

if __name__ == "__main__":
    fire.Fire(main)