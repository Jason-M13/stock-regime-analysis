import numpy as np
import pandas as pd

class MonteCarloSimulator:
    def __init__(self, n_simulations=1000, horizon=30, random_state=42):
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def fit_regime(self, clustered_df, current_cluster, price_col="close", return_col="ret_1"):
        regime_df = clustered_df[clustered_df["cluster"] == current_cluster].copy()

        if regime_df.empty:
            raise ValueError("No rows found for the selected cluster.")

        if return_col not in regime_df.columns:
            raise ValueError(f"Missing return column: {return_col}")

        returns = regime_df[return_col].dropna()

        if len(returns) < 2:
            raise ValueError("Not enough return data in the selected cluster for simulation.")

        self.mu = returns.mean()
        self.sigma = returns.std()
        self.current_cluster = current_cluster
        self.start_price = clustered_df[price_col].iloc[-1]

        return self.mu, self.sigma

    def simulate_paths(self):
        paths = np.zeros((self.horizon + 1, self.n_simulations))
        paths[0] = self.start_price

        for sim in range(self.n_simulations):
            for t in range(1, self.horizon + 1):
                shock = self.rng.normal(self.mu, self.sigma)
                paths[t, sim] = paths[t - 1, sim] * (1 + shock)

        self.paths = paths
        return paths

    def summarize(self):
        final_prices = self.paths[-1]
        summary = {
            "start_price": float(self.start_price),
            "mean_final_price": float(final_prices.mean()),
            "median_final_price": float(np.median(final_prices)),
            "min_final_price": float(final_prices.min()),
            "max_final_price": float(final_prices.max()),
            "probability_gain": float((final_prices > self.start_price).mean()),
            "probability_loss": float((final_prices < self.start_price).mean()),
            "5th_percentile": float(np.percentile(final_prices, 5)),
            "95th_percentile": float(np.percentile(final_prices, 95)),
        }
        return summary