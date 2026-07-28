import pandas as pd
from src.config import cfg
import os
import numpy as np

articles_file = "articles.csv"
customer_file = "customers.csv"
transaction_file = "transactions_train.csv"

# Load data
def load_data():
    df_articles = pd.read_csv(os.path.join(cfg.DATA_DIR,articles_file), dtype = {
        'article_id' : str,
        'product_code':str
    })
    df_customers = pd.read_csv(os.path.join(cfg.DATA_DIR,customer_file), dtype = {
        'customer_id': str
    })
    df_transactions = pd.read_csv(os.path.join(cfg.DATA_DIR,transaction_file),dtype={
    'article_id': str})
    
    return(df_transactions, df_customers, df_articles)

def split_train_test_val(df):
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    df['week'] = df['t_dat'].dt.strftime('%Y-%U')
    df_test = df[df['week']==cfg.TEST_WEEK]
    df_val = df[df['week']==cfg.VALIDATION_WEEK]
    df_train = df[~(df['week'].isin([cfg.VALIDATION_WEEK,cfg.TEST_WEEK]))]
    return(df_test,df_val,df_train)

if __name__ == '__main__':
    df_transactions, df_customers, df_articles = load_data()
    df_test, df_val, df_train = split_train_test_val(df_transactions)
    print('test:', df_test.shape)
    print('val:', df_val.shape)
    print('train:', df_train.shape)
    print('sum:', df_test.shape[0] + df_val.shape[0] + df_train.shape[0])
    print('total:', df_transactions.shape[0])