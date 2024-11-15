import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    
    # Sort data by date
    data = data.sort_values('Date')
    
    # Feature Engineering: Create additional features if needed
    # Example: Moving Average
    data['MA_10'] = data['Adj Close'].rolling(window=10).mean()
    data = data.dropna()
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Adj Close']])
    
    # Prepare training data
    sequence_length = 60  # Number of days to look back
    x_train = []
    y_train = []
    for i in range(sequence_length, len(scaled_data)):
        x_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler

if __name__ == "__main__":
    x_train, y_train, scaler = preprocess_data('data/AAPL_data.csv')
    # Save the scaler for future use
    import joblib
    joblib.dump(scaler, 'models/scaler.save')
    # Save preprocessed data
    np.save('data/x_train.npy', x_train)
    np.save('data/y_train.npy', y_train)
