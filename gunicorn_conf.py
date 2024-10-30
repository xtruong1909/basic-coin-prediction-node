# Gunicorn config variables
loglevel = "info"
errorlog = "-"  # stderr
accesslog = "-"  # stdout
worker_tmp_dir = "/dev/shm"
graceful_timeout = 120
timeout = 30
keepalive = 5
worker_class = "gthread"
workers = 1
threads = 8
bind = "0.0.0.0:9000"
workers = 2  # Giảm số lượng workers để phù hợp với tải thấp hơn của SARIMA
threads = 2
timeout = 300  # Tăng thời gian timeout để xử lý mô hình SARIMA có thể lâu hơn
