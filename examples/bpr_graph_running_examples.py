import pandas as pd
import numpy as np

from recommender_pierre.bpr.BPRGRAPH import BPRGRAPH

np.object = object

train_df = pd.read_csv('./examples/data/train.csv')
train_df = train_df.drop(train_df.columns[2], axis=1)
train_df = train_df[['USER_ID', 'ITEM_ID', 'TRANSACTION_VALUE']]

items_col = 'ITEM_ID'
users_col = 'USER_ID'
ratings_col = 'TRANSACTION_VALUE'

bpr_params = {
    "user_label": users_col,
    "item_label": items_col,
    "transaction_label": ratings_col
}
bpr = BPRGRAPH(**bpr_params)

rec_list = bpr.train_and_produce_rec_list(user_transactions_df=train_df)
rec_list.head(30)
