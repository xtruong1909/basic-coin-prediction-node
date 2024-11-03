import json
from flask import Flask, Response
from model import download_data, format_data, train_model, forecast_price
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

def update_data():
    """Download price data, format data and train model."""
    tokens = ["ETH", "BTC", "BNB", "SOL", "ARB"]
    for token in tokens:
        logging.info(f"Updating data for token: {token}")
        download_data(token)
        format_data(token)
        train_model(token)

def get_token_inference(token):
    return forecast_price.get(token, 0)

@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference for given token."""
    if not token or token not in ["ETH", "BTC", "BNB", "SOL", "ARB"]:
        error_msg = "Token is required" if not token else "Token not supported"
        logging.error(f"Error: {error_msg}")
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')

    try:
        inference = get_token_inference(token)
        logging.info(f"Returning inference for token: {token}")
        return Response(str(inference), status=200)
    except Exception as e:
        logging.error(f"Error generating inference for token {token}: {str(e)}")
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        logging.info("Data update successful")
        return "0"
    except Exception as e:
        logging.error(f"Error updating data: {str(e)}")
        return "1"

if __name__ == "__main__":
    update_data()
    app.run(host="0.0.0.0", port=8011)
