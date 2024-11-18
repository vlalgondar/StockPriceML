from flask import Flask, render_template, request
import pandas as pd
import plotly
import plotly.express as px
import json

app = Flask(__name__)

@app.route('/')
def index():
    # Load predictions
    data = pd.read_csv('data/predictions.csv')
    fig = px.line(data, x='Date', y=['Actual', 'Predicted'], title='Stock Price Prediction')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('index.html', graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)
