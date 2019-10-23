from sklearn.model_selection import KFold, TimeSeriesSplit
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import sys
sys.path.append('./')
from libs.get_params import get_lgb_params
from libs.feature_select import kolmogorov_smirnov, adversarial_del_list
import lightgbm as lgb
import numpy as np
import pandas as pd
import tables
import gc

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


data_folder = '../input/preprocessed_dataset/'
del_list=adversarial_del_list(100)
X = pd.read_hdf(data_folder+'X.hdf', "df", engine='python')
X_test = pd.read_hdf(data_folder+'X_test.hdf', "df", engine='python')
X.drop(del_list, inplace=True, axis=1)
X_test.drop(del_list, inplace=True, axis=1)
"""
data_folder = '../input/adv075/'
X = pd.read_csv(data_folder+'adv075_train.csv', engine='python')
X_test = pd.read_csv(data_folder+'adv075_test.csv', engine='python')
X.drop('TransactionID', inplace=True, axis=1)
X_test.drop('TransactionID', inplace=True, axis=1)
import pdb; pdb.set_trace()
"""
y = pd.read_csv('../input/preprocessed_dataset/y.hdf', 'df', engine='python')
sub = pd.read_csv('../input/dataset/sample_submission.csv')
NFOLDS = 10
folds = KFold(n_splits=NFOLDS)
#folds = TimeSeriesSplit(n_splits=NFOLDS)
params = get_lgb_params()

columns = X.columns
splits = folds.split(X, y)
y_preds = np.zeros(X_test.shape[0])
y_oof = np.zeros(X.shape[0])
score = 0

feature_importances = pd.DataFrame()
feature_importances['feature'] = columns

#import pdb; pdb.set_trace()
 
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)
    
    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
    
    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")
    
    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
    y_preds += clf.predict(X_test) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()
    
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")

sub['isFraud'] = y_preds
sub.to_csv("submission.csv", index=False)

feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)
feature_importances.to_csv('feature_importances.csv')
import pdb; pdb.set_trace()
plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits))
