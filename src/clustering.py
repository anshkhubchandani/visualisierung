#ansh
from sklearn.cluster import KMeans

#lets find this 3 pretty Wein clusters 

def add_kmeans_clusters(df, target_column, exclude_columns=None, n_clusters=3):
    if exclude_columns is None: # machen wir weil bei cars die model_years weggelassen werden beim clustern
        exclude_columns = []

    columns_to_drop = [target_column] + exclude_columns
    features = df.drop(columns=columns_to_drop, errors="ignore")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    df = df.copy()
    df["cluster"] = kmeans.fit_predict(features) + 1

    return df


# def add_kmeans_clusters(df, n_clusters=3):
#     features = df.drop(columns=["class"])

#     kmeans = KMeans(
#         n_clusters=n_clusters,
#         random_state=42,
#         n_init=10
#     )

#     df = df.copy()
#     df["cluster"] = kmeans.fit_predict(features) + 1

#     return df
