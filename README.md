# Stock Market Regime Analysis and Monte Carlo Simulation

## Overview
A Python project that uses **feature engineering, K-means clustering, PCA, and Monte Carlo simulation** to analyze stock behavior. Instead of only predicting whether a stock will go up or down, this project identifies recurring **market regimes** from historical stock data, visualizes them, detects the current regime, and simulates possible future price paths based on similar historical conditions.

The project uses the **Twelve Data API** to fetch stock time series and Bollinger Band data, then applies unsupervised learning and simulation techniques to better understand stock behavior and future risk.

---

## Features
- Fetches historical stock data using the **Twelve Data API**
- Supports flexible intervals:
  - `1day`
  - `1week`
  - `1month`
- Uses up to **10 years** of historical data by default
- Includes feature engineering such as:
  - Returns (1, 3, 6, 12 periods)
  - Rolling volatility
  - Trend indicators using moving average relationships
  - SMA gap
  - Bollinger Band % and width
  - Momentum features
  - Lagged features
- Applies **K-means clustering** to group historical periods into different **market regimes**
- Uses **PCA (Principal Component Analysis)** to reduce feature dimensions and visualize clusters
- Identifies the **latest/current market regime**
- Runs **Monte Carlo simulation** using the historical return behavior of the current regime
- Produces visualizations including:
  - PCA projection of market regimes
  - Market regimes over time
  - Price colored by market regime
  - Monte Carlo simulated price paths
  - Distribution of final simulated prices

---

## Process Summary
1. **Fetch data** from the API (`Time Series`, `Bollinger Bands`)
2. **Merge and preprocess** the data into a single DataFrame
3. **Generate features** such as returns, volatility, momentum, trend, Bollinger Band measures, and lagged features
4. **Clean the dataset** for clustering
5. **Apply K-means clustering** to identify recurring market regimes
6. **Use PCA** to project the feature space into 2 dimensions for visualization
7. **Detect the latest market regime**
8. **Run Monte Carlo simulation** using historical return behavior from the current regime
9. **Visualize** the clustering results and simulated future price outcomes

---

## How to Run the Project

### 1. Open the Twelve Data website
- Create an account or log in
- Navigate to **API Keys**
- Copy your API key

### 2. Open the project in VSCode
- Download the project ZIP or clone the repository
- Open the project folder in **VSCode**

### 3. Create a new file called `.env`
- Make sure the `.env` file is in the project root directory
- Add the following line:

```env
api_key=YOUR_API_KEY_HERE
