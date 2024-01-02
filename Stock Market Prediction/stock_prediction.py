import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Function to download stock data from Yahoo Finance
def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to preprocess stock data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    training_data_len = int(np.ceil(len(scaled_data) * .95))

    train_data = scaled_data[0:int(training_data_len), :]
    return train_data, scaler

# Function to create the LSTM model
def create_model(train_data):
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=1, batch_size=1)

    return model

# Function to make predictions on test data
def make_predictions(model, scaler, data, start_date, end_date):
    test_data = yf.download(data, start=start_date, end=end_date)
    closing_prices = test_data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(closing_prices)

    x_test = []

    for i in range(60, len(scaled_data)):
        x_test.append(scaled_data[i - 60:i, 0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions, test_data

# Function to plot predictions
def plot_predictions(predictions, test_data):
    train = test_data[:len(test_data)-len(predictions)]
    valid = test_data[len(test_data)-len(predictions):]
    valid['Predictions'] = predictions

    plt.figure(figsize=(16, 8))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

# Main function
def main():
    ticker = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2021-01-01'

    # Download stock data
    data = download_stock_data(ticker, start_date, end_date)

    # Preprocess data
    train_data, scaler = preprocess_data(data)

    # Create and train the model
    model = create_model(train_data)

    # Make predictions on test data
    predictions, test_data = make_predictions(model, scaler, ticker, '2021-01-01', '2022-01-01')

    # Plot predictions
    plot_predictions(predictions, test_data)

if _name_ == "_main_":
    main()