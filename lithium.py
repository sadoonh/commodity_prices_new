import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

# Function to load and preprocess data for lithium
def load_and_preprocess_data(ticker="LAC"):
    lithium = yf.Ticker(ticker)
    hist = lithium.history(period="max")
    df = hist[['Close']].copy()
    df.dropna(inplace=True)
    
    # Calculate moving average (30-day)
    df['Moving_Avg'] = df['Close'].rolling(window=30).mean()
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Close'] = scaler.fit_transform(df[['Close']])
    
    return df, scaler

# Function to create sequences for LSTM
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Function to build LSTM model
def build_lstm_model(time_step=60):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Function to train or load model based on flag
def train_or_load_model(X_train, y_train, model_path="lithium_lstm_model.h5", mode='load'):
    if mode == 'load' and os.path.exists(model_path):
        model = load_model(model_path)
        print("Loaded existing model.")
    elif mode == 'train' or not os.path.exists(model_path):
        model = build_lstm_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)
        model.save(model_path)
        print("Trained and saved new model.")
    else:
        raise ValueError("Invalid mode or model path does not exist. Use 'train' to train a new model or 'load' to load an existing one.")
    return model

# Function to make forecasts
def forecast_next_days(model, data, scaler, time_step=60, days=90):
    forecast_input = data[-time_step:].reshape(1, -1)
    forecast_input = forecast_input.reshape((1, time_step, 1))

    forecast = []
    for _ in range(days):
        next_pred = model.predict(forecast_input, verbose=0)
        forecast.append(next_pred[0, 0])
        next_pred_reshaped = np.array([[next_pred[0, 0]]])
        forecast_input = np.append(forecast_input[:, 1:, :], next_pred_reshaped.reshape((1, 1, 1)), axis=1)

    forecast = np.array(forecast).reshape(-1, 1)
    forecast = scaler.inverse_transform(forecast)
    return forecast

# Function to save the forecast data to CSV
def save_forecast_to_csv(df, forecast, output_file="lithium_forecast.csv"):
    # Prepare a DataFrame for forecast data
    forecast_dates = pd.date_range(df.index[-1], periods=len(forecast) + 1, freq='B')[1:]  # Next 90 business days
    forecast_df = pd.DataFrame(data=forecast, index=forecast_dates, columns=['Future Predict'])

    # Concatenate historical and forecast data
    df['Actual'] = scaler.inverse_transform(df[['Close']])
    result_df = pd.concat([df[['Actual']], forecast_df], axis=0)

    # Save to CSV
    result_df.to_csv(output_file)
    print(f"Forecast saved to {output_file}")

# Function to plot results
def plot_results(df, forecast, scaler):
    # Prepare the plotting range (last 12 months and 90-day forecast)
    start_date = df.index[-1] - pd.DateOffset(months=12)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.loc[start_date:].index, scaler.inverse_transform(df[['Close']].values)[-len(df.loc[start_date:]):], label='Historical Data')
    plt.plot(df.loc[start_date:].index, df['Moving_Avg'].loc[start_date:], label='30-Day Moving Average', linestyle='--')
    forecast_dates = pd.date_range(df.index[-1], periods=91, freq='B')[1:]  # Next 90 business days
    plt.plot(forecast_dates, forecast, label='90-Day Forecast', color='red')
    plt.title('Lithium Prices: Historical Data (Last 12 Months), Moving Average, and 90-Day Forecast')
    plt.xlabel('Date')
    plt.ylabel('Lithium Price')
    plt.legend()
    plt.show()

# Main workflow
if __name__ == "__main__":
    df, scaler = load_and_preprocess_data()
    data = df['Close'].values.reshape(-1, 1)

    # Prepare training data
    time_step = 60
    train_size = int(len(data) * 0.8)
    train_data = data[0:train_size, :]

    X_train, y_train = create_sequences(train_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Set mode to 'train' or 'load' based on your requirement
    model = train_or_load_model(X_train, y_train, mode='train')

    # Forecast next 90 days
    forecast = forecast_next_days(model, data, scaler, time_step, days=90)

    # Save forecast data to CSV
    save_forecast_to_csv(df, forecast, output_file="lithium_forecast.csv")

    # Plot results
    plot_results(df, forecast, scaler)

    # Calculate performance metrics for the test set
    X_test, y_test = create_sequences(data[train_size:], time_step)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    test_predict = model.predict(X_test)
    test_predict = scaler.inverse_transform(test_predict)

    mape = mean_absolute_percentage_error(y_test, test_predict)
    rmse = np.sqrt(mean_squared_error(y_test, test_predict))
    r2 = r2_score(y_test, test_predict)

    print(f'MAPE: {mape:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')
