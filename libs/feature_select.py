from tqdm import tqdm
from scipy.stats import ks_2samp
import pandas as pd

def kolmogorov_smirnov(train, test, p_threshold):
#https://upura.hatenablog.com/entry/2019/03/03/233534
    list_p_value=[]
    for i in tqdm(train.columns):
        list_p_value.append(ks_2samp(test[i], train[i])[1])

    Se = pd.Series(list_p_value, index=train.columns).sort_values()
    list_discarded = list(Se[Se < p_threshold].index)
    print("discarded_columns: ", list_discarded)
    train.drop(list_discarded, inplace=True, axis=1)
    test.drop(list_discarded, inplace=True, axis=1)
    return train, test

#def adversarial_del_list():
 #   del_list = ['id_01_count_dist','id_31_count_dist','id_36_count_dist','id_31','version_id_31','id_13','D13','id_36','id_33_count_dist','id_30','V167','TransactionAmt_to_std_card4','TransactionAmt','TransactionAmt_to_mean_card4','TransactionAmt_to_std_addr1','TransactionAmt_to_std_card1','TransactionAmt_to_mean_addr1','TransactionAmt_to_mean_card1','TransactionAmt_Log','addr1__card1','card1_count_full','addr1','card1']
  #  return del_list

def adversarial_del_list(idx):
    feature_imp=pd.read_csv('feature_imp.csv')
    del_list = list(feature_imp['Feature'].values[:idx])
    return del_list
