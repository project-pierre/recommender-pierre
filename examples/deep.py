import pandas as pd
import numpy as np
np.object = object

from recommender_pierre.autoencoders.DeppAutoEncModel import DeppAutoEncModel

# Train Data (subset of all interactions)
df = pd.read_csv('./examples/data/interactions_train_df.csv')
df = df[['user_id', 'content_id', 'game', 'view']]
auto_model = DeppAutoEncModel(
    user_label="user_id", item_label="content_id", transaction_label="view",
    path_model="/home/diego/Code/recommender-pierre/examples/model_params/weights-best-model.hdf5"
)
auto_rec_list_df = auto_model.train_and_produce_rec_list(user_transactions_df=df)
auto_rec_list_df.head(30)