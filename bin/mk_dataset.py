import pandas as pd
import numpy as np
import sys
sys.path.append('./')
from libs.feature_utils import *
from libs.data_utils import load_data


# LOAD DATASET
folder_path = '../input/dataset/'
train_identity, train_transaction, test_identity, test_transaction, sub = load_data(folder_path)

# id_split
train_identity = id_split(train_identity)
test_identity = id_split(test_identity)

# merge data
print('Merging data...')
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
print('Data was successfully merged!\n')
del train_identity, train_transaction, test_identity, test_transaction

print(f'Train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')
print(f'Test dataset has {test.shape[0]} rows and {test.shape[1]} columns.')
gc.collect()

print('drop not useful cols')
train, test = drop_not_useful_cols(train, test)

print('add ave&std')
train, test = add_some_ave_std(train, test)

print('add new features')
train, test = add_new_features(train, test)

print('email feature')
train, test = email_utils(train, test)

print('encode object type')
train, test = object_encode(train, test)

print('save dataset')
X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT'], axis=1)
y = train.sort_values('TransactionDT')['isFraud']
X_test = test.drop(['TransactionDT'], axis=1)

save_path = '../input/preprocessed_dataset/'
X.to_hdf(save_path+'X.hdf','df')
y.to_hdf(save_path+'y.hdf','df')
X_test.to_hdf(save_path+'X_test.hdf','df')

