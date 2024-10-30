import os

# Base paths for application and data storage
app_base_path = os.getenv("APP_BASE_PATH", default=os.getcwd())
data_base_path = os.path.join(app_base_path, "data")

# Model parameters for SARIMA
SARIMA_ORDER = (2, 1, 2)  # Updated to improve model accuracy
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)
TREND = 'c'  # Trend parameter for SARIMA model
MAX_FORECAST_STEPS = 1  # Forecast for the next step

# Paths to store model files
model_dir = os.path.join(data_base_path, "models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_file_path_template = os.path.join(model_dir, "{}_sarima_model.pkl")

# API and logging configurations
BINANCE_API_BASE_URL = "https://fapi.binance.com"
API_RETRY_ATTEMPTS = 5
API_RETRY_DELAY = 1000  # milliseconds
API_RETRY_MAX_DELAY = 10000  # milliseconds
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", default="INFO")

# Thread pool configuration
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))  # Number of threads for concurrent operations
