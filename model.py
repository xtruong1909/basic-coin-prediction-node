import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from config import data_base_path
import random
import requests
import retrying
import joblib
import logging
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dictionary to store forecasted prices for each token
forecast_price = {}

# Path to store Binance data
binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
MAX_DATA_SIZE = 1500  # Limit data size to comply with Binance API constraints
INITIAL_FETCH_SIZE = 1000  # Initial fetch size limited to avoid exceeding Binance API limits

@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def fetch_prices(symbol, interval="1m", limit=1500, start_time=None, end_time=None):
    """
    Fetch historical price data from Binance API.
    
    Parameters:
        symbol (str): Symbol for which data is to be fetched.
        interval (str): Time interval for the data.
        limit (int): Number of data points to fetch (maximum 1500).
        start_time (int): Start time in milliseconds.
        end_time (int): End time in milliseconds.
    
    Returns:
        list: JSON response containing the historical price data.
    """
    try:
        base_url = "https://fapi.binance.com"
        endpoint = f"/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1500)  # Ensure limit does not exceed 1500
        }
        if start_time and end_time and start_time < end_time:
            params['startTime'] = start_time
            params['endTime'] = end_time

        logging.info(f"Fetching data for {symbol} with params: {params}")
        url = base_url + endpoint
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logging.error(f'Failed to fetch prices for {symbol} from Binance API: {str(e)}')
        raise e

def calculate_rsi(df, period=14):
    """
    Calculate the Relative Strength Index (RSI) for the given DataFrame.
    
    Parameters:
        df (DataFrame): DataFrame containing the price data.
        period (int): Period for calculating RSI (default is 14).
    
    Returns:
        DataFrame: DataFrame with an additional 'rsi' column.
    """
    delta = df['close'].diff()  # Calculate the difference between consecutive closing prices
    gain = (delta.where(delta > 0, 0)).fillna(0)  # Positive gains
    loss = (-delta.where(delta < 0, 0)).fillna(0)  # Negative losses

    avg_gain = gain.rolling(window=period).mean()  # Average gain over the period
    avg_loss = loss.rolling(window=period).mean()  # Average loss over the period

    rs = avg_gain / avg_loss  # Relative Strength
    rsi = 100 - (100 / (1 + rs))  # RSI formula
    df['rsi'] = rsi  # Add RSI to DataFrame
    return df

def download_data(token):
    """
    Download historical price data for the given token and save it to CSV.
    
    Parameters:
        token (str): Token for which data is to be downloaded.
    """
    symbols = f"{token.upper()}USDT"
    interval = "5m"  # Time interval for the data
    current_datetime = datetime.now()
    download_path = os.path.join(binance_data_path, token.lower())
    
    file_path = os.path.join(download_path, f"{token.lower()}_5m_data.csv")

    if os.path.exists(file_path):
        # If data already exists, fetch the most recent data
        start_time = int((current_datetime - timedelta(minutes=500)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        if start_time < end_time:  # Ensure start_time is less than end_time
            new_data = fetch_prices(symbols, interval, limit=1500, start_time=start_time, end_time=end_time)
        else:
            logging.warning(f"Invalid start_time ({start_time}) and end_time ({end_time}) for {token}. Skipping data fetch.")
            new_data = []
    else:
        # If no data exists, fetch initial data
        start_time = int((current_datetime - timedelta(minutes=INITIAL_FETCH_SIZE * 5)).timestamp() * 1000)
        end_time = int(current_datetime.timestamp() * 1000)
        if start_time < end_time:  # Ensure start_time is less than end_time
            new_data = fetch_prices(symbols, interval, limit=1500, start_time=start_time, end_time=end_time)
        else:
            logging.warning(f"Invalid start_time ({start_time}) and end_time ({end_time}) for {token}. Skipping data fetch.")
            new_data = []

    if len(new_data) == 0:
        logging.warning(f"No new data fetched for {token}.")
        return

    # Create DataFrame from fetched data
    new_df = pd.DataFrame(new_data, columns=[
        "start_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", 
        "taker_buy_quote_asset_volume", "ignore"
    ])

    if os.path.exists(file_path):
        # Append new data to existing data
        old_df = pd.read_csv(file_path)
        combined_df = pd.concat([old_df, new_df])
        combined_df = combined_df.drop_duplicates(subset=['start_time'], keep='last')
    else:
        combined_df = new_df

    # Limit data size to MAX_DATA_SIZE
    if len(combined_df) > MAX_DATA_SIZE:
        combined_df = combined_df.iloc[-MAX_DATA_SIZE:]

    # Create directory if it does not exist
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    combined_df.to_csv(file_path, index=False)
    logging.info(f"Updated data for {token} saved to {file_path}. Total rows: {len(combined_df)}")

def format_data(token):
    """
    Format the downloaded data for further processing and calculate RSI.
    
    Parameters:
        token (str): Token for which data is to be formatted.
    """
    path = os.path.join(binance_data_path, token.lower())
    file_path = os.path.join(path, f"{token.lower()}_5m_data.csv")

    if not os.path.exists(file_path):
        logging.warning(f"No data file found for {token}")
        return

    df = pd.read_csv(file_path)

    # Select only the required columns
    columns_to_use = [
        "start_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
    ]

    if set(columns_to_use).issubset(df.columns):
        df = df[columns_to_use]
        df.columns = [
            "start_time", "open", "high", "low", "close", "volume",
            "end_time", "quote_asset_volume", "n_trades", 
            "taker_volume", "taker_volume_usd"
        ]
        df.index = pd.to_datetime(df["start_time"], unit='ms')  # Set start_time as index
        df.index.name = "date"

        # Calculate RSI and add it to the DataFrame
        df = calculate_rsi(df)

        output_path = os.path.join(data_base_path, f"{token.lower()}_price_data.csv")
        df.sort_index().to_csv(output_path)
        logging.info(f"Formatted data saved to {output_path}")
    else:
        logging.warning(f"Required columns are missing in {file_path}. Skipping this file.")

def train_model(token):
    """
    Train a SARIMA model on the token's historical data and forecast the next value.
    
    Parameters:
        token (str): Token for which the model is to be trained.
    """
    time_start = datetime.now()

    # Load formatted price data
    price_data = pd.read_csv(os.path.join(data_base_path, f"{token.lower()}_price_data.csv"))
    price_data["date"] = pd.to_datetime(price_data["date"])
    price_data.set_index("date", inplace=True)
    df = price_data.resample('10T').mean()  # Resample data to 10-minute intervals

    df = df.dropna()  # Drop rows with NaN values

    # Define SARIMA parameters
    order = (2, 1, 2)  # Increased order to improve model performance
    seasonal_order = (1, 1, 1, 12)  # P, D, Q, s (assuming yearly seasonality for monthly data)

    try:
        # Train the SARIMA model including volume as an exogenous variable
        model = SARIMAX(df['close'], exog=df[['volume']], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        sarima_model = model.fit(disp=False)

        # Save the trained model
        joblib.dump(sarima_model, os.path.join(data_base_path, f'{token.lower()}_sarima_model.pkl'))

        # Forecast the next value
        forecast_steps = 1
        forecast = sarima_model.get_forecast(steps=forecast_steps, exog=df[['volume']].iloc[-forecast_steps:])
        forecast_mean = forecast.predicted_mean.iloc[-1]

        # Adjust forecasted price based on RSI
        latest_rsi = df['rsi'].iloc[-1]
        
        if latest_rsi > 80:
            adjustment = random.uniform(-0.001 * forecast_mean, 0)  # Giảm từ 0% đến 0.1%
        elif latest_rsi < 20:
            adjustment = random.uniform(0, 0.001 * forecast_mean)  # Tăng từ 0% đến 0.1%
        else:
            adjustment = 0  # Giữ nguyên

        adjusted_price = forecast_mean + adjustment

        # Adjust forecasted price based on volume changes
        if len(df) >= 2:
            volume_change = (df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2]
            price_change = df['close'].iloc[-1] - df['close'].iloc[-2]
            
            if volume_change >= 0.2:
                if price_change > 0:
                    # Volume increased by 20% or more and price increased
                    volume_adjustment = random.uniform(0, 0.01 * adjusted_price)  # Tăng từ 0% đến 1%
                    adjusted_price += volume_adjustment
                elif price_change < 0:
                    # Volume increased by 20% or more and price decreased
                    volume_adjustment = random.uniform(-0.01 * adjusted_price, 0)  # Giảm từ 0% đến 1%
                    adjusted_price += volume_adjustment

        # Store the forecasted price
        forecast_price[token] = adjusted_price

        logging.info(f"Forecasted price for {token}: {forecast_price[token]}")

    except Exception as e:
        logging.error(f"An error occurred while fitting the SARIMA model for {token}: {e}")
        return

    time_end = datetime.now()
    logging.info(f"Time elapsed forecast: {time_end - time_start}")

def update_data():
    """
    Download, format, and train models for a list of tokens.
    """
    tokens = ["ETH", "BTC", "BNB", "SOL", "ARB"]
    with ThreadPoolExecutor(max_workers=len(tokens)) as executor:
        executor.map(lambda token: update_single_token(token), tokens)

def update_single_token(token):
    """
    Update data, format, and train the model for a single token.
    
    Parameters:
        token (str): The token to update.
    """
    try:
        logging.info(f"Updating data for token: {token}")
        download_data(token)
        format_data(token)
        train_model(token)
    except Exception as e:
        logging.error(f"Error updating data for {token}: {e}")

if __name__ == "__main__":
    update_data()
