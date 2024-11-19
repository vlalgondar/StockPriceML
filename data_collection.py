# data_collection.py

import yfinance as yf
import pandas as pd
import os

def download_stock_data(ticker, start_date, end_date):
    print(f"Downloading data for {ticker}")
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    data.reset_index(inplace=True)
    data_folder = 'data'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    data.to_csv(f"{data_folder}/{ticker}_data.csv", index=False)
    print(f"Data saved to {data_folder}/{ticker}_data.csv")

if __name__ == "__main__":
    ticker = 'AAPL'  # Default ticker
    start_date = '2010-01-01'
    end_date = '2023-10-01'
    download_stock_data(ticker, start_date, end_date)
