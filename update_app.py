import os
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

inference_address = os.environ.get("INFERENCE_API_ADDRESS")
if not inference_address:
    logging.error("INFERENCE_API_ADDRESS environment variable is not set.")
    exit(1)

url = f"{inference_address}/update"

logging.info("UPDATING INFERENCE WORKER DATA")

try:
    response = requests.get(url)
    response.raise_for_status()
    content = response.text

    if content == "0":
        logging.info("Response content is '0'")
        exit(0)
    else:
        logging.error("Unexpected response content: %s", content)
        exit(1)
except requests.RequestException as e:
    logging.error(f"Request failed: {e}")
    exit(1)
