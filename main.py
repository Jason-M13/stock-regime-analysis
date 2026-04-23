from client import Client
from dataConversion import merge_tables, build_features, prepare_clustering_data
from predictor import RegimeClusterer
from visuals import Visualizer
from simulation import MonteCarloSimulator

def run(ticker, interval):
    client = Client.configure()

    # Fetch raw data
    b_bands = client.get_bbands(ticker, interval)
    t_series = client.get_time_series(ticker, interval)

    df = merge_tables(t_series, b_bands)
    df = build_features(df)

    # Choose clustering features
    feature_cols = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "vol_6",
    "close_over_sma10", "close_over_sma20",
    "sma_gap_10_20",
    "bb_percentage", "bb_width",
    "momentum_3", "momentum_6",
    "ret_1_lag1", "ret_1_lag2",
    "bb_percentage_lag1", "bb_percentage_lag2"
    ]

    # Clean for clustering
    df = prepare_clustering_data(df, feature_cols)

    # Fit clustering model
    clusterer = RegimeClusterer(feature_cols=feature_cols, n_clusters=4)
    clustered_df = clusterer.fit(df)

    # Latest regime
    latest_cluster, latest_point = clusterer.predict_latest_cluster(clustered_df)
    print(f"Latest cluster/regime: {latest_cluster}")

    simulator = MonteCarloSimulator(n_simulations=1000, horizon=30, random_state=42)
    mu, sigma = simulator.fit_regime(clustered_df, latest_cluster, price_col="close", return_col="ret_1")
    paths = simulator.simulate_paths()
    summary = simulator.summarize()

    print("\nMonte Carlo Summary:")
    for key, value in summary.items():
        if "probability" in key:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value:.2f}")

    # Visualize
    visualizer = Visualizer(clustered_df)
    visualizer.plot_pca_clusters()
    visualizer.plot_cluster_timeline()
    visualizer.plot_price_with_clusters()
    visualizer.plot_monte_carlo_paths(paths)
    visualizer.plot_final_price_histogram(paths)


if __name__ == "__main__":
    print("Options: 1day, 1week, 1month")
    interval = input("Choose an interval: ")
    ticker = input("Choose a stock ticker (e.g., AAPL, MSFT): ")

    run(ticker, interval)