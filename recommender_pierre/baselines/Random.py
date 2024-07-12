import random
from math import ceil

import pandas as pd
from tqdm import tqdm


class RandomRecommender:

    def __init__(
            self,
            list_size: int = 10, batch_size: int = 30,
            user_label: str = "USER_ID", item_label: str = "ITEM_ID",
            transaction_label: str = "TRANSACTION_VALUE"
    ):
        self.user_label = user_label
        self.item_label = item_label
        self.transaction_label = transaction_label
        self.list_size = list_size
        self.batch_size = batch_size
        self.items_ids = []
        self.user_transactions_df = None

    def set_items_ids(self, items_ids):
        self.items_ids = items_ids

    def set_data(self, user_transactions_df):
        self.user_transactions_df = user_transactions_df

    def recommend_for_one(self, user_id, user_profile):
        user_items_ids = user_profile[self.item_label].tolist()
        to_recommend = list(set(self.items_ids) - set(user_items_ids))
        recommendation_ids = random.sample(to_recommend, self.list_size)
        ranking = [1/(i + 1) for i, _ in enumerate(recommendation_ids)]
        return pd.DataFrame(
            {
                self.user_label: user_id,
                self.item_label: recommendation_ids,
                self.transaction_label: ranking
            }
        )

    def recommend_for_batch(self, progress, users_batch=None):
        preds = [
            self.recommend_for_one(
                user_id=user,
                user_profile=self.user_transactions_df[self.user_transactions_df[self.user_label] == user].copy()
            )
            for user in users_batch
        ]
        progress.update(len(users_batch))
        progress.set_description("Recommendation: ")
        return preds

    def recommend_for_all(self, user_transactions_df):
        users_ids = list(user_transactions_df[self.user_label].unique())

        progress = tqdm(total=len(users_ids))
        loops = ceil(len(users_ids)/self.batch_size)

        user_preds = [pd.concat(
            self.recommend_for_batch(
                users_batch=users_ids[i * self.batch_size: (i + 1) * self.batch_size],
                progress=progress
            )
        ) for i in range(0, loops)]
        progress.close()
        return pd.concat(user_preds)

    def train_and_produce_rec_list(self, user_transactions_df):
        user_transactions_df.sort_values(self.user_label, inplace=True)

        self.set_data(user_transactions_df=user_transactions_df)
        self.set_items_ids(items_ids=list(user_transactions_df[self.item_label].unique()))

        return self.recommend_for_all(user_transactions_df=user_transactions_df)
