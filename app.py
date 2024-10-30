import json
import logging
from flask import Flask, Response
from model import download_data, format_data, train_model, forecast_price

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TOKENS = ["ETH", "BTC", "BNB", "SOL", "ARB"]


def update_data():
    """Download price data, format data and train model."""
    for token in TOKENS:
        try:
            logging.info(f"Updating data for token: {token}")
            download_data(token)
            format_data(token)
            train_model(token)
            logging.info(f"Successfully updated data for token: {token}")
        except Exception as e:
            logging.error(f"Error updating data for token {token}: {e}")


def get_token_inference(token):
    """Get forecast price for the given token."""
    try:
        inference = forecast_price.get(token, 0)
        logging.info(f"Inference for token {token}: {inference}")
        return inference
    except Exception as e:
        logging.error(f"Error generating inference for token {token}: {e}")
        return 0


@app.route("/inference/<string:token>")
def generate_inference(token):
    """Generate inference for given token."""
    if not token or token not in TOKENS:
        error_msg = "Token is required" if not token else "Token not supported"
        logging.warning(f"Inference request failed: {error_msg}")
        return Response(json.dumps({"error": error_msg}), status=400, mimetype='application/json')

    try:
        inference = get_token_inference(token)
        return Response(json.dumps({"token": token, "forecast_price": inference}), status=200, mimetype='application/json')
    except Exception as e:
        logging.error(f"Error generating inference: {e}")
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')


@app.route("/update")
def update():
    """Update data and return status."""
    try:
        update_data()
        return Response(json.dumps({"status": "success"}), status=200, mimetype='application/json')
    except Exception as e:
        logging.error(f"Error during update: {e}")
        return Response(json.dumps({"status": "failure", "error": str(e)}), status=500, mimetype='application/json')


if __name__ == "__main__":
    try:
        update_data()
    except Exception as e:
        logging.error(f"Initial data update failed: {e}")
    app.run(host="0.0.0.0", port=8011)
