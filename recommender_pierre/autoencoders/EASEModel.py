from pandas import DataFrame
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count


class EASEModel:
    """
    Code adapted from:
    https://github.com/Darel13712/ease_rec

    Reference:
    Harald Steck. 2019. Embarrassingly Shallow Autoencoders for Sparse Data.
    In The World Wide Web Conference (WWW '19).
    Association for Computing Machinery, New York, NY, USA, 3251–3257.
    https://doi.org/10.1145/3308558.3313710
    """
    def __init__(
            self, lambda_: float = 0.5, implicit=True,
            user_label: str = "USER_ID", item_label: str = "ITEM_ID",
            transaction_label: str = "TRANSACTION_VALUE",
            list_size: int = 10
    ):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()
        self.X = None
        self.B = None
        self.pred = None
        self.lambda_ = lambda_
        self.implicit = implicit
        self.user_label = user_label
        self.item_label = item_label
        self.transaction_label = transaction_label
        self.recommendations = None
        self.list_size = list_size

    def _get_users_and_items(self, interactions_df: DataFrame) -> tuple:
        """

        :param interactions_df: A pandas Dataframe with the users' interactions.
        :return: A tuple with two lists: users id and items id.
        """
        users = self.user_enc.fit_transform(interactions_df.loc[:, self.user_label])
        items = self.item_enc.fit_transform(interactions_df.loc[:, self.item_label])
        return users, items

    def fit(self, interactions_df: DataFrame):
        """
        interactions_df: pandas.DataFrame with columns user_id, item_id and (rating)
        lambda_: l2-regularization term
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        """
        users, items = self._get_users_and_items(interactions_df=interactions_df)
        values = (
            np.ones(interactions_df.shape[0])
            if self.implicit
            else interactions_df[
                     self.transaction_label].to_numpy() / interactions_df[
                self.transaction_label].max()
        )

        self.X = csr_matrix((values, (users, items)))

        G = self.X.T.dot(self.X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += self.lambda_
        P = np.linalg.inv(G)
        self.B = P / (-np.diag(P))
        self.B[diagIndices] = 0

        self.pred = self.X.dot(self.B)

    def predict(self, train: DataFrame, users: list, items: list, k: int = 10):
        items = self.item_enc.transform(items)
        dd = train.loc[train[self.user_label].isin(users)]
        dd['ci'] = self.item_enc.transform(dd[self.item_label])
        dd['cu'] = self.user_enc.transform(dd[self.user_label])
        g = dd.groupby('cu')
        with Pool(cpu_count()) as p:
            user_preds = p.starmap(
                self.predict_for_user,
                [(user, group, self.pred[user, :], items, k) for user, group in g],
            )
        self.recommendations = pd.concat(user_preds)
        self.recommendations[self.item_label] = self.item_enc.inverse_transform(
            self.recommendations[self.item_label])
        self.recommendations[self.user_label] = self.user_enc.inverse_transform(
            self.recommendations[self.user_label])
        return self.recommendations

    def predict_for_user(self, user: str, group: DataFrame, pred: DataFrame, items: list, k: int):
        watched = set(group['ci'])
        candidates = [item for item in items if item not in watched]
        pred = np.take(pred, candidates)
        res = np.argpartition(pred, -k)[-k:]
        r = pd.DataFrame(
            {
                self.user_label: [user] * len(res),
                self.item_label: np.take(candidates, res),
                self.transaction_label: np.take(pred, res),
            }
        ).sort_values(self.transaction_label, ascending=False)
        return r

    def train_and_produce_rec_list(self, user_transactions_df):
        self.fit(interactions_df=user_transactions_df)
        users_ids = user_transactions_df[self.user_label].unique().tolist()
        items_ids = user_transactions_df[self.item_label].unique().tolist()
        return self.predict(
            train=user_transactions_df, users=users_ids, items=items_ids, k=self.list_size
        )
