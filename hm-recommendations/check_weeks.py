from src.data_loader import load_data, split_train_test_val

df_transactions, df_customers, df_articles = load_data()
df_test, df_val, df_train = split_train_test_val(df_transactions)

print("VAL week daily counts:")
print(df_val.groupby(df_val['t_dat'].dt.date).size())

print("\nTEST week daily counts:")
print(df_test.groupby(df_test['t_dat'].dt.date).size())
