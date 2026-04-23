import pandas as pd

def returns(table, windows=[1, 3, 6, 12], price_col="close"):
    try:
        if price_col not in table.columns:
            raise KeyError(f"Column '{price_col}' not found in DataFrame")

        for i in windows:
            table[f"ret_{i}"] = table[price_col] / table[price_col].shift(i) - 1

    except Exception as e:
        print(f"[returns] Error: {e}")
    return table

def volatility(table, ret_col="ret_1", window=6):
    try:
        if ret_col not in table.columns:
            raise KeyError(f"Column '{ret_col}' not found in DataFrame")

        table[f"vol_{window}"] = table[ret_col].rolling(window).std()

    except Exception as e:
        print(f"[volatility] Error: {e}")
    return table

def trend_indicator(table, price_col="close", windows=[10, 20]):
    try:
        if price_col not in table.columns:
            raise KeyError(f"Column '{price_col}' not found in DataFrame")

        for i in windows:
            sma = table[price_col].rolling(i).mean()
            table[f"sma_{i}"] = sma
            table[f"close_over_sma{i}"] = table[price_col] / sma

        if len(windows) >= 2:
            short_w = windows[0]
            long_w = windows[1]
            table[f"sma_gap_{short_w}_{long_w}"] = (
                table[f"sma_{short_w}"] / table[f"sma_{long_w}"] - 1
            )

    except Exception as e:
        print(f"[trend_indicator] Error: {e}")
    return table

def bbands_feature(table, upper="upper_band", middle="middle_band", lower="lower_band", price_col="close"):
    try:
        missing_cols = [c for c in [upper, middle, lower, price_col] if c not in table.columns]
        if missing_cols:
            raise KeyError(f"Missing columns for Bollinger Bands: {missing_cols}")

        denominator = table[upper] - table[lower]
        if (denominator == 0).any():
            print("[bbands_feature] Warning: Zero denominator encountered")

        table["bb_percentage"] = (table[price_col] - table[lower]) / denominator
        table["bb_width"] = denominator / table[middle]

    except Exception as e:
        print(f"[bbands_feature] Error: {e}")
    return table

def check_momentum(table, price_col="close", momenta=[3, 6]):
    try:
        if price_col not in table.columns:
            raise KeyError(f"Column '{price_col}' not found in DataFrame")

        for i in momenta:
            table[f"momentum_{i}"] = table[price_col] / table[price_col].shift(i) - 1

    except Exception as e:
        print(f"[check_momentum] Error: {e}")
    return table

def add_lags(table, cols=("ret_1", "bb_percentage", "adx"), ks=(1, 2)):
    try:
        for col in cols:
            if col not in table.columns:
                print(f"[add_lags] Warning: Column '{col}' not found, skipping.")
                continue

            for k in ks:
                table[f"{col}_lag{k}"] = table[col].shift(k)

    except Exception as e:
        print(f"[add_lags] Error: {e}")
    return table