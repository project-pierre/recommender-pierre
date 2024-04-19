import pandas as pd
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .BaseModel import BaseModel


class DeppAutoEncModel(BaseModel):
    """
    create model
    Reference:
    KUCHAIEV, Oleksii; GINSBURG, Boris. Training deep autoencoders for collaborative filtering.
        arXiv preprint arXiv:1708.01715, 2017.
            https://github.com/NVIDIA/DeepRecommender
            https://arxiv.org/pdf/1708.01715.pdf
    """

    def __init__(
            self,
            factors: int = 15, epochs: int = 100, batch: int = 64, activation: str = 'selu', dropout: float = 0.1,
            lr: float = 0.0001, reg: float = 0.01, loss: str = 'mse', validation_split: float = 0.3,
            list_size: int = 10,
            user_label: str = "USER_ID", item_label: str = "ITEM_ID", transaction_label: str = "TRANSACTION_VALUE",
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
        """
        Autoencoder for Collaborative Filter Model
        """

        # Input
        input_layer = Input(shape=(X.shape[1],), name='UserScore')

        # Encoder
        # -----------------------------
        enc = Dense(512, activation=self.activation, name='EncLayer1')(input_layer)

        # Latent Space
        # -----------------------------
        lat_space = Dense(256, activation=self.activation, name='LatentSpace')(enc)
        lat_space = Dropout(self.dropout, name='Dropout')(lat_space)  # Dropout

        # Decoder
        # -----------------------------
        dec = Dense(512, activation=self.activation, name='DecLayer1')(lat_space)

        # Output
        output_layer = Dense(X.shape[1], activation='linear', name='UserScorePred')(dec)

        # This model maps an input to its reconstruction
        return Model(input_layer, output_layer)

    def fit(self, X, y):
        # Build model
        model = self.build_model(X)

        model.compile(optimizer=Adam(lr=self.lr), loss=self.loss)  # 'mean_absolute_error'

        # train
        hist = model.fit(
            x=X, y=y,
            epochs=self.epochs, batch_size=self.batch, shuffle=True,
            validation_split=self.validation_split, callbacks=self.callbacks_list(path_model=self.path_model)
        )

        model.load_weights(self.path_model)
        self.model = model

        return model, hist

    def predict(self, X):
        # Predict
        pred = self.model.predict(X)

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
