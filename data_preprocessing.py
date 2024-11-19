# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import os

def preprocess_data(file_path, ticker):
    data = pd.read_csv(file_path)
    data = data.dropna()
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    # Sort data by date
    data = data.sort_values('Date')

    # Feature Engineering
    data['MA_10'] = data['Adj Close'].rolling(window=10).mean()
    data = data.dropna()

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Adj Close']])

    # Prepare training data
    sequence_length = 60
    x_train = []
    y_train = []
    for i in range(sequence_length, len(scaled_data)):
        x_train.append(scaled_data[i - sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Save the scaler for this ticker
    scaler_folder = 'models'
    if not os.path.exists(scaler_folder):
        os.makedirs(scaler_folder)
    joblib.dump(scaler, f'{scaler_folder}/{ticker}_scaler.save')
    print(f"Scaler saved to {scaler_folder}/{ticker}_scaler.save")

    # Save preprocessed data (optional)
    # np.save(f'data/{ticker}_x_train.npy', x_train)
    # np.save(f'data/{ticker}_y_train.npy', y_train)

    return x_train, y_train, scaler

if __name__ == "__main__":
    x_train, y_train, scaler = preprocess_data('data/AAPL_data.csv', 'AAPL')
