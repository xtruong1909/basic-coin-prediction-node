import os
import requests
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

inference_address = os.environ.get("INFERENCE_API_ADDRESS", "http://inference:8011")
url = f"{inference_address}/update"

logging.info("UPDATING INFERENCE WORKER DATA")

try:
    response = requests.get(url)
    if response.status_code == 200:
        # Request was successful
        content = response.text

        if content == "0":
            logging.info("Response content is '0' - Update successful")
            exit(0)
        else:
            logging.error("Unexpected response content")
            exit(1)
    else:
        # Request failed
        logging.error(f"Request failed with status code: {response.status_code}")
        exit(1)
except requests.RequestException as e:
    logging.error(f"Request exception occurred: {e}")
    exit(1)
