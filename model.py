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

forecast_price = {}

binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
MAX_DATA_SIZE = 1000  # Giới hạn số lượng dữ liệu tối đa khi lưu trữ
INITIAL_FETCH_SIZE = 1000  # Số lượng nến lần đầu tải về
COINGECKO_API_KEY = "CG-8MtvACYdTwpB32DpjhLgeeVb"

@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def fetch_prices(token, days=1, interval="minute"):  # interval can be "minute" or "hour"
    try:
        base_url = "https://api.coingecko.com/api/v3/coins"
        endpoint = f"/{token}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": interval
        }
        headers = {
            "Authorization": f"Bearer {COINGECKO_API_KEY}"
        }

        url = base_url + endpoint
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["prices"]
    except Exception as e:
        print(f'Failed to fetch prices for {token} from CoinGecko API: {str(e)}')
        raise e

def download_data(token):
    interval = "minute"
    days = 1
    current_datetime = datetime.now()
    download_path = os.path.join(binance_data_path, token.lower())
    
    file_path = os.path.join(download_path, f"{token.lower()}_5m_data.csv")

    if os.path.exists(file_path):
        new_data = fetch_prices(token, days, interval)
    else:
        new_data = fetch_prices(token, days, interval)

    new_df = pd.DataFrame(new_data, columns=["timestamp", "close"])
    new_df["start_time"] = pd.to_datetime(new_df["timestamp"], unit='ms')

    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        combined_df = pd.concat([old_df, new_df])
        combined_df = combined_df.drop_duplicates(subset=['start_time'], keep='last')
    else:
        combined_df = new_df

    if len(combined_df) > MAX_DATA_SIZE:
        combined_df = combined_df.iloc[-MAX_DATA_SIZE:]

    if not os.path.exists(download_path):
        os.makedirs(download_path)
    combined_df.to_csv(file_path, index=False)
    print(f"Updated data for {token} saved to {file_path}. Total rows: {len(combined_df)}")

def format_data(token):
    path = os.path.join(binance_data_path, token.lower())
    file_path = os.path.join(path, f"{token.lower()}_5m_data.csv")

    if not os.path.exists(file_path):
        print(f"No data file found for {token}")
        return

    df = pd.read_csv(file_path)

    columns_to_use = [
        "start_time", "close"
    ]

    if set(columns_to_use).issubset(df.columns):
        df = df[columns_to_use]
        df.columns = [
            "start_time", "close"
        ]
        df.index = pd.to_datetime(df["start_time"], unit='ms')
        df.index.name = "date"

        output_path = os.path.join(data_base_path, f"{token.lower()}_price_data.csv")
        df.sort_index().to_csv(output_path)
        print(f"Formatted data saved to {output_path}")
    else:
        print(f"Required columns are missing in {file_path}. Skipping this file.")

def train_model(token):
    time_start = datetime.now()

    price_data = pd.read_csv(os.path.join(data_base_path, f"{token.lower()}_price_data.csv"))
    price_data["date"] = pd.to_datetime(price_data["date"])
    price_data.set_index("date", inplace=True)
    df = price_data.resample('10T').mean()

    df = df.dropna()

    # Define SARIMA parameters
    order = (1, 1, 1)  # p, d, q
    seasonal_order = (1, 1, 1, 12)  # P, D, Q, s (assuming yearly seasonality for monthly data)

    try:
        model = SARIMAX(df['close'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        sarima_model = model.fit(disp=False)
    except Exception as e:
        raise RuntimeError(f"An error occurred while fitting the SARIMA model: {e}")

    # Save the model
    joblib.dump(sarima_model, f'{token.lower()}_sarima_model.pkl')

    # Forecasting next value
    forecast_steps = 1
    forecast = sarima_model.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean.iloc[-1]

    # Xử lý biên độ ngẫu nhiên
    now = datetime.now().astimezone()  # Giờ địa phương
    is_weekend = now.weekday() >= 5  # Thứ 7, Chủ nhật
    is_night = now.hour >= 19 or now.hour < 7  # Từ 19 giờ đến 7 giờ sáng

    adjusted_price = calculate_weighted_average_price(forecast_mean, token, is_weekend, is_night)

    # Điều chỉnh giá dự đoán cuối cùng
    avg_previous_price = df['close'].mean()
    if avg_previous_price < forecast_mean:
        adjustment = random.uniform(-0.0005 * forecast_mean, 0)  # Giảm từ 0% đến 0.05%
    else:
        adjustment = random.uniform(0, 0.0005 * forecast_mean)  # Tăng từ 0% đến 0.05%
    adjusted_price += adjustment

    forecast_price[token] = adjusted_price

    print(f"Forecasted price for {token}: {forecast_price[token]}")

    time_end = datetime.now()
    print(f"Time elapsed forecast: {time_end - time_start}")

def calculate_weighted_average_price(predicted_price, token, is_weekend, is_night):
    path = os.path.join(data_base_path, f"{token.lower()}_price_data.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.resample('10T').mean()

    if is_night:
        if is_weekend:
            recent_prices = df['close'].dropna().iloc[-5:]
            weights = [0.6, 0.1, 0.1, 0.1, 0.1]
        else:
            recent_prices = df['close'].dropna().iloc[-3:]
            weights = [0.7, 0.15, 0.15]
        
        if len(recent_prices) < len(weights):
            return predicted_price

        weighted_sum = sum(w * p for w, p in zip(weights, recent_prices))
        adjusted_price = (0.4 * predicted_price) + weighted_sum
    else:
        adjusted_price = predicted_price

    return adjusted_price

def update_data():
    tokens = ["ethereum", "bitcoin", "binancecoin", "solana", "arbitrum"]
    for token in tokens:
        download_data(token)
        format_data(token)
        train_model(token)

if __name__ == "__main__":
    update_data()
