import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Embedding, Flatten, Dropout
from keras.layers import add
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from .BaseModel import BaseModel


class CDAEModel(BaseModel):
    """
    create model

    Reference:
    Yao Wu, Christopher DuBois, Alice X. Zheng, and Martin Ester. 2016.
    Collaborative Denoising Auto-Encoders for Top-N Recommender Systems.
    In Proceedings of the Ninth ACM International Conference on Web Search and Data Mining
    (WSDM '16).
    Association for Computing Machinery, New York, NY, USA, 153–162.
    https://doi.org/10.1145/2835776.2835837
    """

    def __init__(
            self,
            factors: int = 15, epochs: int = 100, batch: int = 64, activation: str = 'selu',
            dropout: float = 0.1,
            lr: float = 0.0001, reg: float = 0.01, loss: str = 'mse', validation_split: float = 0.3,
            list_size: int = 10,
            user_label: str = "USER_ID", item_label: str = "ITEM_ID",
            transaction_label: str = "TRANSACTION_VALUE",
            path_model: str = "/tmp/weights-best-model.keras"
    ):
        self.factors = factors
        self.epochs = epochs
        self.batch = batch
        self.activation = activation
        self.dropout = dropout
        self.lr = lr
        self.reg = reg
        self.loss = loss
        self.validation_split = validation_split
        self.path_model = path_model
        self.model = None
        self.user_item_matrix = None
        self.interact_matrix = None
        self.user_label = user_label
        self.item_label = item_label
        self.transaction_label = transaction_label
        self.list_size = list_size

    def build_model(self, X):
        # Input
        x_item = Input((X.shape[1],), name='UserScore')

        h_item = Dropout(self.dropout)(x_item)
        h_item = Dense(
            self.factors, kernel_regularizer=l2(self.reg), bias_regularizer=l2(self.reg), activation=self.activation,
            name="Dense_1"
        )(h_item)
        # dtype should be int to connect to Embedding layer
        x_user = Input((1,), name='UserContent')
        h_user = Embedding(
            len(np.unique(list(self.user_item_matrix.index))) + 1, self.factors, input_length=1,
            embeddings_regularizer=l2(self.reg), name="Embedding"
        )(x_user)
        h_user = Flatten()(h_user)

        h = add([h_item, h_user], name='LatentSpace')
        y = Dense(X.shape[1], activation='linear', name='UserScorePred')(h)

        return Model(inputs=[x_item, x_user], outputs=y, name="Model_Final")

    def fit(self, X, y):
        # Build model
        model = self.build_model(X)

        model.compile(optimizer=Adam(learning_rate=self.lr), loss=self.loss)  # 'mean_absolute_error'
        users_ids = list(self.user_item_matrix.index)
        # train
        hist = model.fit(
            x=[X, np.array(users_ids).reshape(len(users_ids), 1)], y=y,
            epochs=self.epochs, batch_size=self.batch, shuffle=True,
            validation_split=self.validation_split, callbacks=self.callbacks_list(path_model=self.path_model)
        )

        model.load_weights(self.path_model)
        self.model = model

        return model, hist

    def predict(self, X):
        # Predict
        users_ids = list(self.user_item_matrix.index)
        pred = self.model.predict([X, np.array(users_ids).reshape(len(users_ids), 1)])

        # remove watched items from predictions
        pred = pred * (X == 0)

        return pred

    def recommender_for_user(self, user_id):
        """
        Recommender for one user
        """
        pred_scores = self.interact_matrix.loc[user_id].values

        df_scores = pd.DataFrame({
            self.item_label: list(self.user_item_matrix.columns),
            self.transaction_label: pred_scores
        })

        df_rec = df_scores.set_index(self.item_label) \
            .sort_values(self.transaction_label, ascending=False) \
            .head(self.list_size)

        df_rec[self.user_label] = user_id
        df_rec.reset_index(inplace=True)

        return df_rec[df_rec[self.transaction_label] > 0]

    def train_and_produce_rec_list(self, user_transactions_df):
        self.user_item_matrix = user_transactions_df.pivot(
            index=self.user_label, columns=self.item_label, values=self.transaction_label).fillna(0)

        X = self.user_item_matrix.values
        y = self.user_item_matrix.values

        _, _ = self.fit(X, y)

        new_matrix = self.predict(X) * (X == 0)

        self.interact_matrix = pd.DataFrame(
            new_matrix, columns=self.user_item_matrix.columns, index=self.user_item_matrix.index
        )

        rec_list_df = list(map(self.recommender_for_user, list(self.user_item_matrix.index)))
        return pd.concat(rec_list_df)
