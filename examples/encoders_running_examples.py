import pandas as pd
import numpy as np
np.object = object

from recommender_pierre.autoencoders.DeppAutoEncModel import DeppAutoEncModel
from recommender_pierre.autoencoders.CDAEModel import CDAEModel
from recommender_pierre.autoencoders.EASEModel import EASEModel

# Train Data (subset of all interactions)
df = pd.read_csv('./examples/data/interactions_train_df.csv')
df = df[['user_id', 'content_id', 'game', 'view']]

cdae_model = CDAEModel(
    user_label="user_id", item_label="content_id", transaction_label="view",
    path_model="/home/diego/Code/recommender-pierre/examples/model_params/weights-best-model.keras"
)
cdae_rec_list_df = cdae_model.train_and_produce_rec_list(user_transactions_df=df)
cdae_rec_list_df.head(30)

auto_model = DeppAutoEncModel(
    user_label="user_id", item_label="content_id", transaction_label="view",
    path_model="/home/diego/Code/recommender-pierre/examples/model_params/weights-best-model.keras"
)
auto_rec_list_df = auto_model.train_and_produce_rec_list(user_transactions_df=df)
auto_rec_list_df.head(30)

auto_model = EASEModel(
    lambda_=0.5,
    user_label="user_id", item_label="content_id", transaction_label="view"
)
ease_rec_list_df = auto_model.train_and_produce_rec_list(user_transactions_df=df)
ease_rec_list_df.head(30)
