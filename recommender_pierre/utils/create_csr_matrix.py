from scipy.sparse import csr_matrix


def create_matrix(data, users_col, items_col, ratings_col):
    """
    creates the sparse user-item interaction matrix,
    if the data is not in the format where the interaction only
    contains the positive items (indicated by 1), then use the
    threshold parameter to determine which items are considered positive

    Parameters
    ----------
    data : DataFrame
        implicit rating data

    users_col : str
        user column name

    items_col : str
        item column name

    ratings_col : str
        implicit rating column name

    Returns
    -------
    ratings : scipy sparse csr_matrix, shape [n_users, n_items]
        user/item ratings matrix

    data : DataFrame
        implict rating data that retains only the positive feedback
        (if specified to do so)
    """
    # this ensures each user has at least 2 records to construct a valid
    # train and test split in downstream process, note we might purge
    # some users completely during this process
    data_user_num_items = (data
                           .groupby(users_col)
                           .agg(**{'num_items': (items_col, 'count')})
                           .reset_index())
    data = data.merge(data_user_num_items, on=users_col, how='inner')
    data = data[data['num_items'] > 1]

    for col in (items_col, users_col, ratings_col):
        data[col] = data[col].astype('category')

    ratings = csr_matrix((data[ratings_col],
                          (data[users_col].cat.codes, data[items_col].cat.codes)))
    ratings.eliminate_zeros()
    return ratings, data
