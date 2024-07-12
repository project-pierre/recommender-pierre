import pandas as pd

from recommender_pierre.baselines.Random import RandomRecommender
from recommender_pierre.baselines.Popularity import PopularityRecommender

train_df = pd.read_csv('./examples/data/train.csv')
train_df = train_df.drop(train_df.columns[2], axis=1)
train_df = train_df[['USER_ID', 'ITEM_ID', 'TRANSACTION_VALUE']]

items_col = 'ITEM_ID'
users_col = 'USER_ID'
ratings_col = 'TRANSACTION_VALUE'

params = {
    "user_label": users_col,
    "item_label": items_col,
    "transaction_label": ratings_col
}
rd = RandomRecommender(**params)

rec_list = rd.train_and_produce_rec_list(user_transactions_df=train_df)
rec_list.head(30)

pop = PopularityRecommender(**params)

pop_rec_list = pop.train_and_produce_rec_list(user_transactions_df=train_df)
pop_rec_list.head(30)
