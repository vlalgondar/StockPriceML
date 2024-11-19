# model_prediction.py

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt
import joblib
import os

def predict_stock_price(model_file, scaler_file, data_file, ticker):
    # Load the model and scaler
    model = tf.keras.models.load_model(model_file)
    scaler = joblib.load(scaler_file)

    # Load and preprocess data
    data = pd.read_csv(data_file)
    data = data.dropna()
    actual_prices = data['Adj Close'].values
    total_dataset = data[['Adj Close']].values

    # Prepare test inputs
    sequence_length = 60
    inputs = total_dataset[-(sequence_length + len(actual_prices)):]
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(sequence_length, len(inputs)):
        X_test.append(inputs[i - sequence_length:i, 0])

    X_test = np.array(X_test)
    if X_test.size != 0:
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    else:
        print("X_test is empty. Please check the input data and preprocessing steps.")
        return

    # Get predictions
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Adjust actual_prices to match predicted_prices length
    actual_prices = actual_prices[-len(predicted_prices):]

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(actual_prices, predicted_prices))
    print(f"Root Mean Squared Error for {ticker}: {rmse}")

    # Save predictions
    predictions = pd.DataFrame({
        'Date': data['Date'].values[-len(predicted_prices):],
        'Actual': actual_prices,
        'Predicted': predicted_prices.flatten()
    })
    predictions_file = f'data/{ticker}_predictions.csv'
    predictions.to_csv(predictions_file, index=False)
    print(f"Predictions saved to {predictions_file}")

if __name__ == "__main__":
    predict_stock_price('models/stock_prediction_model.h5', 'models/scaler.save', 'data/AAPL_data.csv', 'AAPL')
