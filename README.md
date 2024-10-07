CAC 40 Stock Price Prediction Using LSTM
This project aims to predict the stock prices of the CAC 40 index using a Long Short-Term Memory (LSTM) neural network model. The CAC 40 is a benchmark French stock market index representing the 40 largest companies listed on the Euronext Paris exchange. Predicting stock trends helps analysts and investors make informed decisions based on historical data.

Project Overview
Objective: To develop a machine learning model that forecasts CAC 40 stock prices using historical data.
Data Source: Stock data is sourced from Yahoo Finance using the yfinance Python library.
Tools: Python, TensorFlow, yfinance, Scikit-Learn, Pandas, NumPy, Matplotlib
Methodology
Data Collection: The project uses Yahoo Finance to retrieve daily historical data for the CAC 40 index.
Data Preprocessing: The data is scaled and transformed to be suitable for input into an LSTM neural network.
Model Training: An LSTM model is built and trained on the data to recognize and predict stock price trends.
Evaluation: The model is evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE), and the predictions are visualized against actual prices.
Installation
To run this project locally, follow these steps:

Clone the Repository:
bash

git clone https://github.com/your-username/CAC40-Stock-Prediction.git
Navigate to the Project Directory:
bash

cd CAC40-Stock-Prediction
Install the Required Libraries:
bash

pip install -r requirements.txt
Usage
Run the Python script to train the model and generate predictions:
bash

python stock_prediction.py
Modify the start and end dates in the script if you want to change the prediction period.
Results
The plot below shows the actual vs. predicted stock prices, showcasing the model's ability to capture general trends.


Future Enhancements
Potential improvements include incorporating additional technical indicators, experimenting with different model architectures, or deploying the model as a web application.
