import os

# Set base path for the application
app_base_path = os.getenv("APP_BASE_PATH", default=os.getcwd())

# Define data base path for storing data
data_base_path = os.path.join(app_base_path, "data")

# Define model directory for SARIMA model files
model_directory = os.path.join(data_base_path, "models")

# Environment configurations for worker and model
worker_loop_seconds = int(os.getenv("WORKER_LOOP_SECONDS", 60))
model_name = os.getenv("MODEL_NAME", "sarima")
