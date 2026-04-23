import os
import requests
import pandas as pd
from dotenv import load_dotenv

class Client:
    @classmethod
    def configure(cls):
        load_dotenv()
        api_key = os.getenv("api_key")
        return cls(api_key)

    def __init__(self, api_key: str, base_url: str = "https://api.twelvedata.com"):
        if not api_key:
            raise ValueError("No API Key")
        self.api_key = api_key
        self.base_url = base_url

    def _extract_values(self, response, endpoint_name):
        if "values" not in response:
            message = response.get("message", "Unknown API error")
            raise ValueError(f"[{endpoint_name}] API response missing 'values': {message}")
        return response["values"]

    def _apply_cutoff(self, df, years=10):
        cutoff = pd.Timestamp.today() - pd.DateOffset(years=years)
        return df.loc[df.index >= cutoff]

    def get_adx(self, ticker_symbol, interval):
        url = f"{self.base_url}/adx?symbol={ticker_symbol}&interval={interval}&outputsize=5000&apikey={self.api_key}"
        response = requests.get(url).json()

        values = self._extract_values(response, "ADX")
        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["adx"] = pd.to_numeric(df["adx"], errors="coerce")

        df = df.sort_values("datetime").set_index("datetime")
        df = self._apply_cutoff(df, years=10)

        return df

    def get_bbands(self, ticker_symbol, interval):
        url = f"{self.base_url}/bbands?symbol={ticker_symbol}&interval={interval}&time_period=20&sd=2&outputsize=5000&apikey={self.api_key}"
        response = requests.get(url).json()

        values = self._extract_values(response, "BBANDS")
        df = pd.DataFrame(values)

        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ["upper_band", "middle_band", "lower_band"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("datetime").set_index("datetime")
        df = self._apply_cutoff(df, years=10)

        return df

    def get_time_series(self, ticker_symbol, interval):
        url = f"{self.base_url}/time_series?symbol={ticker_symbol}&interval={interval}&outputsize=5000&apikey={self.api_key}"
        response = requests.get(url).json()

        values = self._extract_values(response, "TIME_SERIES")
        df = pd.DataFrame(values)

        df["datetime"] = pd.to_datetime(df["datetime"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("datetime").set_index("datetime")
        df = self._apply_cutoff(df, years=10)

        return df