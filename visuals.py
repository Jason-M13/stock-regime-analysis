import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, clustered_df):
        self.df = clustered_df.copy()

    def plot_pca_clusters(self):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            self.df["pca_1"],
            self.df["pca_2"],
            c=self.df["cluster"],
            alpha=0.7
        )
        plt.title("PCA Projection of Market Regimes")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.show()

    def plot_cluster_timeline(self):
        plt.figure(figsize=(12, 4))
        plt.scatter(self.df.index, self.df["cluster"], alpha=0.7)
        plt.title("Market Regimes Over Time")
        plt.xlabel("Date")
        plt.ylabel("Cluster")
        plt.tight_layout()
        plt.show()

    def plot_price_with_clusters(self, price_col="close"):
        plt.figure(figsize=(12, 5))
        plt.scatter(
            self.df.index,
            self.df[price_col],
            c=self.df["cluster"],
            alpha=0.8
        )
        plt.title("Price Colored by Market Regime")
        plt.xlabel("Date")
        plt.ylabel(price_col.capitalize())
        plt.tight_layout()
        plt.show()

    def plot_monte_carlo_paths(self, paths):
        plt.figure(figsize=(10, 6))
        plt.plot(paths, alpha=0.1)
        plt.title("Monte Carlo Simulated Price Paths")
        plt.xlabel("Time Step")
        plt.ylabel("Price")
        plt.tight_layout()
        plt.show()

    def plot_final_price_histogram(self, paths):
        final_prices = paths[-1]

        plt.figure(figsize=(10, 6))
        plt.hist(final_prices, bins=30)
        plt.title("Distribution of Final Simulated Prices")
        plt.xlabel("Final Price")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()