from flask import Flask, render_template, request
import pandas as pd
import plotly
import plotly.express as px
import json
import os

# Import your data processing and prediction functions
from data_collection import download_stock_data
from data_preprocessing import preprocess_data
from model_prediction import predict_stock_price

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        # Paths for the data and model files
        data_file = f'data/{ticker}_data.csv'
        model_file = 'models/stock_prediction_model.h5'
        scaler_file = 'models/scaler.save'
        predictions_file = f'data/{ticker}_predictions.csv'

        try:
            # Step 1: Download data for the new ticker
            download_stock_data(ticker, '2010-01-01', '2023-10-01')

            # Step 2: Preprocess the data
            x_train, y_train, scaler = preprocess_data(data_file, ticker)

            # Step 3: Predict stock prices
            predict_stock_price(model_file, scaler_file, data_file, ticker)

            # Step 4: Load predictions
            predictions = pd.read_csv(predictions_file)
            fig = px.line(predictions, x='Date', y=['Actual', 'Predicted'], title=f'{ticker} Stock Price Prediction')
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            return render_template('index.html', graphJSON=graphJSON, ticker=ticker)

        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")
            return render_template('index.html', error=f"Could not process ticker {ticker}. Please try again.", ticker='AAPL')

    else:
        # Default ticker
        ticker = 'AAPL'
        # Load default predictions
        predictions_file = f'data/{ticker}_predictions.csv'
        if not os.path.exists(predictions_file):
            # If predictions don't exist, generate them
            download_stock_data(ticker, '2010-01-01', '2023-10-01')
            x_train, y_train, scaler = preprocess_data(f'data/{ticker}_data.csv', ticker)
            predict_stock_price('models/stock_prediction_model.h5', 'models/scaler.save', f'data/{ticker}_data.csv', ticker)
        predictions = pd.read_csv(predictions_file)
        fig = px.line(predictions, x='Date', y=['Actual', 'Predicted'], title=f'{ticker} Stock Price Prediction')
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('index.html', graphJSON=graphJSON, ticker=ticker)

if __name__ == '__main__':
    app.run(debug=True)
