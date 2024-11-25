# from math import ceil
#
# import tensorflow as tf
# # import tensorflow.compat.v1 as tf
# # tf.disable_v2_behavior()
# import numpy as np
# import pandas as pd
# import scipy.sparse as sp
#
# from tqdm import tqdm
#
#
# class BPRGRAPH(object):
#     def __init__(
#             self,
#             iterations: int = 50, batch_size: int = 30, factors: int = 64, learning_rate: float = 0.005,
#             samples: int = 15000,
#             lambda_user: float = 0.0000001, lambda_item: float = 0.0000001, lambda_bias: float = 0.0000001,
#             user_label: str = "USER_ID", item_label: str = "ITEM_ID",
#             transaction_label: str = "TRANSACTION_VALUE", list_size=10
#
#     ):
#         self.lambda_item = lambda_item
#         self.lambda_user = lambda_user
#         self.lambda_bias = lambda_bias
#         self.iterations = iterations
#         self.factors = factors
#         self.batch_size = batch_size
#         self.samples = samples
#         self.learning_rate = learning_rate
#         self.user_label = user_label
#         self.item_label = item_label
#         self.user_label_look = user_label + "_id"
#         self.item_label_look = item_label + "_id"
#         self.transaction_label = transaction_label
#         self.list_size = list_size
#         # Set up our Tensorflow graph
#         self.graph = tf.Graph()
#         self.init = None
#         self.step = None
#         self.loss = None
#         self.session = None
#         self.u_auc = None
#         self.u = None
#         self.i = None
#         self.j = None
#
#     def init_variable(self, size, dim, name=None):
#         '''
#         Helper function to initialize a new variable with
#         uniform random values.
#         '''
#         std = np.sqrt(2 / dim)
#         return tf.Variable(tf.random.uniform([size, dim], -std, std), name=name)
#
#     def embed(self, inputs, size, dim, name=None):
#         '''
#         Helper function to get a Tensorflow variable and create
#         an embedding lookup to map our user and item
#         indices to vectors.
#         '''
#         emb = self.init_variable(size, dim, name)
#         return tf.nn.embedding_lookup(emb, inputs)
#
#     def get_variable(self, graph, session, name):
#         '''
#         Helper function to get the value of a
#         Tensorflow variable by name.
#         '''
#         v = graph.get_operation_by_name(name)
#         v = v.values()[0]
#         v = v.eval(session=session)
#         return v
#
#     def preparing(self, users: list, items: list):
#         with self.graph.as_default():
#             '''
#             Loss function:
#             -SUM ln σ(xui - xuj) + λ(w1)**2 + λ(w2)**2 + λ(w3)**2 ...
#             ln = the natural log
#             σ(xuij) = the sigmoid function of xuij.
#             λ = lambda regularization value.
#             ||W||**2 = the squared L2 norm of our model parameters.
#
#             '''
#
#             # Input into our model, in this case our user (u),
#             # known item (i) an unknown item (i) triplets.
#             self.u = tf.compat.v1.placeholder(tf.int32, shape=(None, 1))
#             self.i = tf.compat.v1.placeholder(tf.int32, shape=(None, 1))
#             self.j = tf.compat.v1.placeholder(tf.int32, shape=(None, 1))
#
#             # User feature embedding
#             u_factors = self.embed(self.u, len(users), self.factors, 'user_factors')  # U matrix
#
#             # Known and unknown item embeddings
#             item_factors = self.init_variable(len(items), self.factors, "item_factors")  # V matrix
#             i_factors = tf.nn.embedding_lookup(item_factors, self.i)
#             j_factors = tf.nn.embedding_lookup(item_factors, self.j)
#
#             # i and j bias embeddings.
#             item_bias = self.init_variable(len(items), 1, "item_bias")
#             i_bias = tf.nn.embedding_lookup(item_bias, self.i)
#             i_bias = tf.reshape(i_bias, [-1, 1])
#             j_bias = tf.nn.embedding_lookup(item_bias, self.j)
#             j_bias = tf.reshape(j_bias, [-1, 1])
#
#             # Calculate the dot product + bias for known and unknown
#             # item to get xui and xuj.
#             xui = i_bias + tf.reduce_sum(u_factors * i_factors, axis=2)
#             xuj = j_bias + tf.reduce_sum(u_factors * j_factors, axis=2)
#
#             # We calculate xuij.
#             xuij = xui - xuj
#
#             # Calculate the mean AUC (area under curve).
#             # if xuij is greater than 0, that means that
#             # xui is greater than xuj (and thats what we want).
#             self.u_auc = tf.reduce_mean(tf.cast(xuij > 0, tf.float32))
#
#             # Output the AUC value to tensorboard for monitoring.
#             tf.summary.scalar('auc', self.u_auc)
#
#             # Calculate the squared L2 norm ||W||**2 multiplied by λ.
#             l2_norm = tf.add_n([
#                 self.lambda_user * tf.reduce_sum(tf.multiply(u_factors, u_factors)),
#                 self.lambda_item * tf.reduce_sum(tf.multiply(i_factors, i_factors)),
#                 self.lambda_item * tf.reduce_sum(tf.multiply(j_factors, j_factors)),
#                 self.lambda_bias * tf.reduce_sum(tf.multiply(i_bias, i_bias)),
#                 self.lambda_bias * tf.reduce_sum(tf.multiply(j_bias, j_bias))
#             ])
#
#             # Calculate the loss as ||W||**2 - ln σ(Xuij)
#             # loss = l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(xuij)))
#             self.loss = -tf.reduce_mean(tf.math.log(tf.sigmoid(xuij))) + l2_norm
#
#             # Train using the Adam optimizer to minimize
#             # our loss function.
#             # opt = tf.optimizers.Adam(learning_rate=self.learning_rate)
#             # self.step = opt.minimize(self.loss, var_list=None, tape=tf.GradientTape(persistent=True))
#             opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
#             self.step = opt.minimize(self.loss)
#
#             # Initialize all tensorflow variables.
#             self.init = tf.compat.v1.global_variables_initializer()
#
#     def training(self, items, uids, iids):
#         # ------------------
#         # GRAPH EXECUTION
#         # ------------------
#
#         # Run the session.
#         self.session = tf.compat.v1.Session(config=None, graph=self.graph)
#         self.session.run(self.init)
#
#         # This has noting to do with tensorflow but gives
#         # us a nice progress bar for the training.
#         progress = tqdm(total=self.batch_size * self.iterations)
#         auc = None
#         l = None
#         for _ in range(self.iterations):
#             for _ in range(self.batch_size):
#                 # We want to sample one known and one unknown
#                 # item for each user.
#
#                 # First we sample 15000 uniform indices.
#                 idx = np.random.randint(low=0, high=len(uids), size=self.samples)
#
#                 # We then grab the users matching those indices.
#                 batch_u = uids[idx].reshape(-1, 1)
#
#                 # Then the known items for those users.
#                 batch_i = iids[idx].reshape(-1, 1)
#
#                 # Lastly we randomly sample one unknown item for each user.
#                 batch_j = np.random.randint(
#                     low=0, high=len(items), size=(self.samples, 1), dtype='int32')
#
#                 # Feed our users, known and unknown items to
#                 # our tensorflow graph.
#                 feed_dict = {self.u: batch_u, self.i: batch_i, self.j: batch_j}
#
#                 # We run the session.
#                 _, l, auc = self.session.run([self.step, self.loss, self.u_auc], feed_dict)
#
#             progress.update(self.batch_size)
#             progress.set_description('Loss: %.3f | AUC: %.3f' % (l, auc))
#
#         progress.close()
#
#     # ---------------------
#     # MAKE RECOMMENDATION
#     # ---------------------
#
#     def make_recommendation(self, progress, item_lookup, user_id=None, num_items=10):
#         """
#         Recommend items for a given user given a trained model
#         Args:
#             user_id (int): The id of the user we want to create recommendations for.
#             num_items (int): How many recommendations we want to return.
#         Returns:
#             recommendations (pandas.DataFrame): DataFrame with num_items artist names and scores
#         """
#
#         # Grab our user matrix U
#         user_vecs = self.get_variable(self.graph, self.session, 'user_factors')
#
#         # Grab our item matrix V
#         item_vecs = self.get_variable(self.graph, self.session, 'item_factors')
#
#         # Grab our item bias
#         item_bi = self.get_variable(self.graph, self.session, 'item_bias').reshape(-1)
#
#         # Calculate the score for our user for all items.
#         rec_vector = np.add(user_vecs[user_id, :].dot(item_vecs.T), item_bi)
#
#         # Grab the indices of the top users
#         item_idx = np.argsort(rec_vector)[::-1][:num_items]
#
#         # Map the indices to artist names and add to dataframe along with scores.
#         artists, scores = [], []
#         user = []
#         for idx in item_idx:
#             user.append(item_lookup[self.user_label].loc[item_lookup[self.user_label_look] == user_id].iloc[0])
#             artists.append(item_lookup[self.item_label].loc[item_lookup[self.item_label_look] == idx].iloc[0])
#             scores.append(rec_vector[idx])
#
#         recommendations = pd.DataFrame({
#             self.user_label: user,
#             self.item_label: artists,
#             self.transaction_label: scores
#         })
#         return recommendations
#
#     def recommend_for_batch(self, progress, item_lookup, users_batch=None, num_items=10):
#         preds = [
#             self.make_recommendation(item_lookup=item_lookup, user_id=user, num_items=num_items, progress=progress)
#             for user in users_batch
#         ]
#         progress.update(len(users_batch))
#         progress.set_description("Recommendation: ")
#         return preds
#
#     def recommend_for_all(self, item_lookup, users, k=10):
#         progress = tqdm(total=len(users))
#         loops = ceil(len(users)/self.batch_size)
#
#         # user_preds = [pd.concat([
#         #     self.make_recommendation(item_lookup=item_lookup, user_id=user, num_items=k, progress=progress)
#         #     for user in users[i * self.batch_size: (i + 1) * self.batch_size]
#         # ]) for i in range(0, loops)]
#
#         user_preds = [pd.concat(
#             self.recommend_for_batch(
#                 item_lookup=item_lookup, users_batch=users[i * self.batch_size: (i + 1) * self.batch_size],
#                 num_items=k, progress=progress
#             )
#         ) for i in range(0, loops)]
#         progress.close()
#         return pd.concat(user_preds)
#
#     def train_and_produce_rec_list(self, user_transactions_df):
#         user_transactions_df.sort_values(self.user_label, inplace=True)
#
#         # Drop any rows with 0 plays
#         user_trans_df = user_transactions_df.loc[user_transactions_df[self.transaction_label] != 0]
#
#         user_trans_df[self.user_label_look] = user_trans_df[self.user_label].astype("category").cat.codes
#         user_trans_df[self.item_label_look] = user_trans_df[self.item_label].astype("category").cat.codes
#
#
#         # Create a lookup frame so we can get the artist
#         # names back in readable form later.
#         item_lookup = user_trans_df[
#             [self.item_label_look, self.item_label,
#              self.user_label_look, self.user_label]
#         ].drop_duplicates()
#         item_lookup[self.item_label_look] = item_lookup[self.item_label_look].astype(int)
#         item_lookup[self.user_label_look] = item_lookup[self.user_label_look].astype(int)
#
#         users = list(user_trans_df[self.user_label_look].unique())
#         items = list(user_trans_df[self.item_label_look].unique())
#         transactions = list(user_trans_df[self.transaction_label])
#
#         # Get the rows and columns for our new matrix
#         rows = user_trans_df[self.user_label_look].tolist()
#         cols = user_trans_df[self.item_label_look].tolist()
#
#         data_sparse = sp.csr_matrix(
#             (
#                 transactions,
#                 (rows, cols)
#             ), shape=(len(users), len(items))
#         )
#
#         uids, iids = data_sparse.nonzero()
#
#         self.preparing(users=users, items=items)
#         self.training(uids=uids, items=items, iids=iids)
#
#         return self.recommend_for_all(
#             item_lookup=item_lookup, users=users, k=self.list_size
#         )
