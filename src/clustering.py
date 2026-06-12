#ansh
from sklearn.cluster import KMeans

#lets find this 3 pretty Wein clusters 

def add_kmeans_clusters(df, n_clusters=3):
    features = df.drop(columns=["class"])

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    df = df.copy()
    df["cluster"] = kmeans.fit_predict(features) + 1

    return df
