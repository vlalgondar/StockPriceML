import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def train_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model.fit(x_train, y_train, epochs=10, batch_size=32)
    
    model.save('models/stock_prediction_model.h5')

    print("Model saved to models/stock_prediction_model.h5")

if __name__ == "__main__":
    x_train = np.load('data/x_train.npy')
    y_train = np.load('data/y_train.npy')
    train_model(x_train, y_train)
