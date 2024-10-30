import os
import requests
import logging
import time

# Set up logging
logging_level = os.environ.get("LOGGING_LEVEL", "INFO").upper()
logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')

inference_address = os.environ.get("INFERENCE_API_ADDRESS")
if not inference_address:
    logging.error("INFERENCE_API_ADDRESS environment variable is not set.")
    exit(1)

url = f"{inference_address}/update"

logging.info("Starting inference worker data update loop")

while True:
    logging.info("UPDATING INFERENCE WORKER DATA")
    try:
        response = requests.get(url)
        response.raise_for_status()
        content = response.text

        if content == "0":
            logging.info("Update successful. Response content is '0'")
        else:
            logging.error("Unexpected response content: %s", content)
    except requests.RequestException as e:
        logging.error(f"Request failed: {e}")

    # Wait before next update
    time.sleep(600)  # Update every 10 minutes
