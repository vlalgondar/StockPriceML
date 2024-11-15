import yfinance as yf
import pandas as pd

def download_stock_data(ticker, start_date, end_date):
    print(f"Downloading data for {ticker}")
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data.to_csv(f"data/{ticker}_data.csv", index=False)
    print(f"Data saved to data/{ticker}_data.csv")

if __name__ == "__main__":
    import os
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    # Parameters
    ticker = 'AAPL'  # You can change this to any ticker symbol
    start_date = '2010-01-01'
    end_date = '2023-10-01'
    download_stock_data(ticker, start_date, end_date)
