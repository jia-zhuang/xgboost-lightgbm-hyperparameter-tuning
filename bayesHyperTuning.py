from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import gc
from datetime import datetime
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import fire

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    else:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != np.object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('Mem. usage decrease to {:5.2f}Mb ({:.1f}% reduction)'.format(end_mem, 100*(start_mem - end_mem)/start_mem))
    return df


def lgb_cv(num_leaves, min_data_in_leaf, bagging_fraction, feature_fraction, lambda_l1, data, target, feature_name, categorical_feature):
    folds = KFold(n_splits=4, shuffle=True, random_state=11)
    oof = np.zeros(data.shape[0])
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(data, target)):
        print(f'fold: {fold_}')
        trn_data = lgb.Dataset(data[trn_idx], label=target[trn_idx], feature_name=feature_name, categorical_feature=categorical_feature)
        val_data = lgb.Dataset(data[val_idx], label=target[val_idx], feature_name=feature_name, categorical_feature=categorical_feature)
        param = {
            # general parameters
            'objective': 'regression',
            'boosting': 'gbdt',
            'metric': 'rmse',
            'learning_rate': 0.005,
            # tuning parameters
            'num_leaves': int(num_leaves),
            'min_data_in_leaf': int(min_data_in_leaf),
            'bagging_freq': 1,
            'bagging_fraction': bagging_fraction,
            'feature_fraction': feature_fraction,
            'lambda_l1': lambda_l1
        }
        clf = lgb.train(param, trn_data, 10000, valid_sets=[trn_data, val_data], verbose_eval=200, early_stopping_rounds=600)
        oof[val_idx] = clf.predict(data[val_idx], num_iteration=clf.best_iteration)
        del clf, trn_idx, val_idx
        gc.collect()
    return -mean_squared_error(target, oof)**0.5

def xgb_cv(max_depth, min_child_weight, subsample, colsample_bytree, data, target):
    folds = KFold(n_splits=4, shuffle=True, random_state=11)
    oof = np.zeros(data.shape[0])
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(data, target)):
        print(f'fold: {fold_}')
        trn_data = xgb.DMatrix(data[trn_idx], label=target[trn_idx])
        val_data = xgb.DMatrix(data[val_idx], label=target[val_idx])
        param = {
            # general parameters
            'objective': 'reg:linear',
            'booster': 'gbtree',
            'eval_metric': 'rmse',
            'learning_rate': 0.005,
            'silent': 1,
            # tuning parameters
            'max_depth': int(max_depth),
            'min_child_weight': min_child_weight,
             #'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree
        }
        clf = xgb.train(param, trn_data, 10000, evals=[(trn_data, 'train'), (val_data, 'valid')], verbose_eval=200, early_stopping_rounds=600)
        oof[val_idx] = clf.predict(val_data)
        del clf, trn_idx, val_idx
        gc.collect()
    return -mean_squared_error(target, oof)**0.5


def optimize_lgb(data, target, feature_name='auto', categorical_feature='auto'):
    def lgb_crossval(num_leaves, min_data_in_leaf, bagging_fraction, feature_fraction, lambda_l1):
        return lgb_cv(num_leaves, min_data_in_leaf, bagging_fraction, feature_fraction, lambda_l1, data, target, feature_name, categorical_feature)
    
    optimizer = BayesianOptimization(lgb_crossval, {
        'num_leaves': (20, 200),
        'min_data_in_leaf': (10, 150),
        'bagging_fraction': (0.5, 1.0),
        'feature_fraction': (0.5, 1.0),
        'lambda_l1': (0, 10)
    })

    start_time = timer()
    optimizer.maximize(init_points=5, n_iter=100, acq='ucb', kappa=10)
    timer(start_time)
    print("Final result:", optimizer.max)


def optimize_xgb(data, target):
    def xgb_crossval(max_depth, min_child_weight, subsample, colsample_bytree):
        return xgb_cv(max_depth, min_child_weight, subsample, colsample_bytree, data, target)
    
    optimizer = BayesianOptimization(xgb_crossval, {
        'max_depth': (4, 10),
        'min_child_weight': (10, 150),
        'subsample': (0.7, 1.0),
        'colsample_bytree': (0.7, 1.0)
    })

    start_time = timer()
    optimizer.maximize(init_points=2, n_iter=20)
    timer(start_time)
    print("Final result:", optimizer.max)
    

def main(train_csv, target_txt):
    train = pd.read_csv(train_csv, index_col=0)
    target = np.loadtxt(target_txt)
    train = reduce_mem_usage(train)
    feats = [col for col in train.columns if col not in ['first_active_month', 'card_id']]
    cat_feats = [col for col in feats if col.startswith('feature_')]
    optimize_lgb(train[feats].values, target, feature_name=feats, categorical_feature=cat_feats)
    #optimize_xgb(train[feats].values, target)

if __name__ == "__main__":
    fire.Fire(main)