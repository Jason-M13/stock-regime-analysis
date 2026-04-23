# Stock Trend Predictor 

## Overview 
A Python project that uses **stock indicators and machine learning** to predict whether a stock’s price will go **Up** or **Down** in the next time interval, along with a **confidence rate** and **test accuracy**. The project uses data from [Twelve Data API](https://twelvedata.com/) to fetch stock data and applies stock features + logistic regression for predictions. The results can be **visualized** to check accuracy on past historical data.  

---

## Features
- Fetches historical stock data (time series, Bollinger Bands, ADX) via **Twelve Data API**.  
- Supports flexible intervals (`1day`, `1week`, `1month`) and history ranges (default: **10 years**).
- **(Note: It is possible to test interval at 1 minute with the input of **1 min**, however be mindful of the API credit usage)**
- Feature engineering includes:
  - Returns (1, 3, 6, 12 periods)  
  - Rolling volatility  
  - Trend indicators (moving averages)  
  - Bollinger Band % and width  
  - Momentum features  
  - Lagged features  
- Converts stock movement into a **binary classification problem** (`Up = 1`, `Down = 0`).  
- Trains a **Logistic Regression model** using `scikit-learn`.  
- Produces predictions with a **confidence rate (probability)**.  
- Includes a **scatter plot** to show backtest results (confidence vs actual outcomes).  

---

## Process Summary 
1. **Fetch data** from the API (`ADX`, `Bollinger Bands`, `Time Series`).  
2. **Merge & preprocess** into a single DataFrame.  
3. **Generate features** (returns, volatility, momentum, bands, lags).  
4. **Convert to binary target** (`1` if next close > current close, else `0`).  
5. **Split into train/test sets**.  
6. **Train Logistic Regression** on training set.  
7. **Predict test set** and calculate probabilities.  
8. **Visualize** model confidence vs actual outcomes.

---

## How to Run the Project
 ### 1. Open the TwelveData website
  - create an account or log in with an existing account
  - Navigate to **API Keys** on the left hand side menu 
  - Click the reveal key and copy it

  ### 2. Open the Project in VSCode
  - Click on **code** and click on download zip
  - Unzip the folder and launch **VSCode**.
  - Load the project into the IDE.

 ### 3. Create a new file called **.env**
  - Ensure the **.env** file is in the same directory as the project 
  - Type out **api_key=** and paste the api key from **step 1** 
  - Make sure there are no spaces

  ### 4. Run main.py
  - Open main.py and allow files to load
  - Simply run main.py

  ### 5. Inputs
  **(Note: Don't add any spaces when entering input into the console, otherwise the data cannot be retrieved)** 
  - Enter the interval options (`1day`, `1week`, `1month`)
  - Enter the desired ticker
 
---

## Requirements & Tools Used 
- Python 3.9+  
- Packages:
- `pandas`
- `numpy`
- `requests`
- `matplotlib`
- `scikit-learn`
- `python-dotenv`

---

## Sources 
This project uses the data from the **Twelve Data API**. Details, including ADX, Bollinger Bands, and Time Series. The model predictions were compared against historical stock movements (as verified via Yahoo Finance).
- **Stock Data API:** [Twelve Data](https://twelvedata.com/)
- **Machine Learning:** [Logistic Regression](https://www.w3schools.com/python/python_ml_logistic_regression.asp)
- **Binary Classifaction:** [Kaggle](https://www.kaggle.com/code/ryanholbrook/binary-classification)
- **Time Series Analysis:** [Kaggle](https://www.kaggle.com/code/prashant111/complete-guide-on-time-series-analysis-in-python)
