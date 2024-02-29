import pandas as pd
import numpy as np
np.object = object

from recommender_pierre.autoencoders.AutoEncModel import AutoEncModel
from recommender_pierre.autoencoders.CDAEModel import CDAEModel

# Train Data (subset of all interactions)
df = pd.read_csv('./examples/data/interactions_train_df.csv')
df = df[['user_id', 'content_id', 'game', 'view']]

cdae_model = CDAEModel(
    user_label="user_id", item_label="content_id",
    transaction_label="view", path_model="/home/diego/Code/recommender-pierre/examples/data/weights-best-model.hdf5"
)
cdae_rec_list_df = cdae_model.train_and_produce_rec_list(user_transactions_df=df)
cdae_rec_list_df.head(30)

auto_model = AutoEncModel(
    user_label="user_id", item_label="content_id",
    transaction_label="view", path_model="/home/diego/Code/recommender-pierre/examples/data/weights-best-model.hdf5"
)
auto_rec_list_df = auto_model.train_and_produce_rec_list(user_transactions_df=df)
auto_rec_list_df.head(30)
