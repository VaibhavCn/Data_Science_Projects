from src.data_loader import load_data, split_train_test_val

df_transactions, df_customers, df_articles = load_data()

df_test, df_val, df_train = split_train_test_val(df_transactions)
print('test:', df_test.shape)
print('val:', df_val.shape)
print('train:', df_train.shape)
print('sum:', df_test.shape[0] + df_val.shape[0] + df_train.shape[0])
print('total:', df_transactions.shape[0])
