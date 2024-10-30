import os  # Thêm dòng này để sử dụng os.getenv()

# Gunicorn config variables
loglevel = "info"
errorlog = "-"  # stderr
accesslog = "-"  # stdout
worker_tmp_dir = "/dev/shm"
graceful_timeout = 120
timeout = 100  # Increased timeout to handle longer SARIMA model processing
keepalive = 5
worker_class = "gthread"
workers = int(os.getenv("WORKERS", "2"))  # Adjust based on environment variable
threads = int(os.getenv("THREADS", "4"))  # Adjust based on environment variable
bind = "0.0.0.0:9000"
preload_app = True  # Preload the application to reduce worker startup time
max_requests = 1000  # Limit the number of requests a worker will handle before restarting
max_requests_jitter = 100  # Add jitter to avoid workers restarting simultaneously
