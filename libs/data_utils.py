import pandas as pd

def load_data(folder_path):
    print('Loading data...')

    train_identity = pd.read_csv(f'{folder_path}train_identity.csv', index_col='TransactionID', engine='python')
    print('\tSuccessfully loaded train_identity!')

    train_transaction = pd.read_csv(f'{folder_path}train_transaction.csv', index_col='TransactionID')
    print('\tSuccessfully loaded train_transaction!')

    test_identity = pd.read_csv(f'{folder_path}test_identity.csv', index_col='TransactionID')
    print('\tSuccessfully loaded test_identity!')

    test_transaction = pd.read_csv(f'{folder_path}test_transaction.csv', index_col='TransactionID')
    print('\tSuccessfully loaded test_transaction!')

    sub = pd.read_csv(f'{folder_path}sample_submission.csv')
    print('\tSuccessfully loaded sample_submission!')

    print('Data was successfully loaded!\n')

    return train_identity, train_transaction, test_identity, test_transaction, sub
