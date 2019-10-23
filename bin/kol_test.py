import pandas as pd
import tables
import sys
sys.path.append('./')
from libs.feature_select import kolmogorov_smirnov

data_dir = '../input/preprocessed_dataset/'
train = pd.read_hdf(data_dir+'X.hdf', 'df')
test = pd.read_hdf(data_dir+'X_test.hdf', 'df')

train, test = kolmogorov_smirnov(train, test, 0.01)
import pdb; pdb.set_trace()
