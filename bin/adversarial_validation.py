import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import model_selection, preprocessing, metrics
import sys
sys.path.append('./')
from libs.feature_select import adversarial_del_list

data_dir = '../input/preprocessed_dataset/'
train = pd.read_hdf(data_dir+'X.hdf','df', nrows=50000)
test = pd.read_hdf(data_dir+'X_test.hdf','df',nrows=50000)
del_list=[]
del_list=adversarial_del_list(420)
train.drop(del_list, inplace=True, axis=1)
test.drop(del_list, inplace=True, axis=1)
import pdb; pdb.set_trace()
use_columns = train.columns.values

train['target']=0
test['target']=1

train = pd.concat([train, test], axis=0)
target = train['target'].values
del train['target'], test

train, val, t_train, t_val = train_test_split(train, target, test_size=0.2, random_state=42)

param = {'num_leaves': 200,
         'min_data_in_leaf': 60, 
         'objective':'binary',
         'max_depth': -1,
         'learning_rate': 0.1,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 17,
         "metric": 'auc',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "n_jobs":-1}

train = lgb.Dataset(train.values, label=t_train)
val = lgb.Dataset(val.values, label=t_val)

num_round = 1000
clf = lgb.train(param, train, num_round, valid_sets = [train, val], verbose_eval=10, early_stopping_rounds = 25)

feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(), use_columns), reverse=True), columns=['Value','Feature'])
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(30))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances-01.png')
#feature_imp = feature_imp.sort_values(by="Value", ascending=False)
#feature_imp.to_csv('feature_imp.csv',index=False)
