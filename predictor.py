import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

class RegimeClusterer:
    def __init__(self, feature_cols, n_clusters=4, random_state=42):
        self.feature_cols = feature_cols
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.pca = PCA(n_components=2)

    def fit(self, df):
        data = df.copy()

        x = data[self.feature_cols].copy()
        x = x.replace([np.inf, -np.inf], np.nan).dropna()

        # keep aligned dataframe
        data = data.loc[x.index].copy()

        x_scaled = self.scaler.fit_transform(x)
        clusters = self.kmeans.fit_predict(x_scaled)
        pca_components = self.pca.fit_transform(x_scaled)

        data["cluster"] = clusters
        data["pca_1"] = pca_components[:, 0]
        data["pca_2"] = pca_components[:, 1]

        self.x = x
        self.x_scaled = x_scaled
        self.clustered_df = data

        self.inertia = self.kmeans.inertia_
        self.silhouette = silhouette_score(x_scaled, clusters)

        print(f"Clusters fitted: {self.n_clusters}")
        print(f"Silhouette Score: {self.silhouette:.4f}")

        return data

    def predict_latest_cluster(self, df):
        latest = df[self.feature_cols].iloc[[-1]].copy()
        latest = latest.replace([np.inf, -np.inf], np.nan).dropna()

        if latest.empty:
            raise ValueError("Latest row contains invalid or missing feature values.")

        latest_scaled = self.scaler.transform(latest)
        cluster = self.kmeans.predict(latest_scaled)[0]
        pca_point = self.pca.transform(latest_scaled)[0]

        print("\nLatest Market Regime:")
        print(f"Cluster: {cluster}")

        return cluster, pca_point