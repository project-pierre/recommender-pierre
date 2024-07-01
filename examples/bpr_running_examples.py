import pandas as pd
import numpy as np

from recommender_pierre.bpr.BPRKNN import BPRKNN

np.object = object

train_df = pd.read_csv('./examples/data/interactions_train_df.csv')
train_df = train_df[['user_id', 'content_id', 'view']]

items_col = 'content_id'
users_col = 'user_id'
ratings_col = 'view'
threshold = 0

bpr_params = {
    'regularization': 0.01,
    'learning_rate': 0.1,
    'iterations': 160,
    'factors': 15,
    'batch_size': 100,
    "user_label": users_col,
    "item_label": items_col,
    "transaction_label": ratings_col
}
bpr = BPRKNN(**bpr_params)

rec_list = bpr.train_and_produce_rec_list(user_transactions_df=train_df)
rec_list.head(30)
