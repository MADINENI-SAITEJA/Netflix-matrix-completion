import numpy as np
import pandas as pd


def load_movielens_fold(base_path, test_path):

    df_train = pd.read_csv(
        base_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    df_test = pd.read_csv(
        test_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    n_users = max(df_train["user_id"].max(),
                  df_test["user_id"].max())

    n_items = max(df_train["item_id"].max(),
                  df_test["item_id"].max())

    train = np.zeros((n_users, n_items))
    test = np.zeros((n_users, n_items))

    for row in df_train.itertuples():
        train[row.user_id - 1, row.item_id - 1] = row.rating

    for row in df_test.itertuples():
        test[row.user_id - 1, row.item_id - 1] = row.rating

    return train, test
