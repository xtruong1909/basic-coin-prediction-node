import os

app_base_path = os.getenv("APP_BASE_PATH", default=os.getcwd())
data_base_path = os.path.join(app_base_path, "data")
# model_file_path = os.path.join(data_base_path, "model.pkl")
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)
MAX_FORECAST_STEPS = 1  # Dự báo cho một bước tiếp theo
