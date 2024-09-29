import os
import json

# Đảm bảo biến môi trường được thiết lập để chỉ đến tệp config.json
config_file_path = os.getenv('CONFIG_FILE', 'config.json')

# Đảm bảo tệp config.json tồn tại và có thể đọc được
if not os.path.exists(config_file_path):
    raise FileNotFoundError(f"Tệp config.json không tìm thấy tại: {config_file_path}")

# Đọc tệp config.json
with open(config_file_path, 'r') as file:
    config = json.load(file)

# Đặt biến data_base_path để sử dụng trong đoạn code của bạn
data_base_path = config.get('data_base_path', 'default/path')
