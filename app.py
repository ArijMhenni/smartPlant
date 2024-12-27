from flask import Flask, jsonify, render_template
import paho.mqtt.client as mqtt
import numpy as np
import tensorflow as tf
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# MQTT Configuration
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
DATA_TOPIC = "plant_health/data"
PREDICTION_TOPIC = "plant_health/prediction"

# TensorFlow Lite model and scaling parameters
MODEL_PATH = "model3.tflite"
FEATURE_MIN = np.array([10.0007236, 18.00199267, 15.00371007, 40.02875752, 200.61548181])
FEATURE_MAX = np.array([39.99316429, 29.9908861, 24.99592885, 69.96887071, 999.8562615])
CLASS_MAPPING = {0: "Healthy", 1: "High Stress", 2: "Moderate Stress"}

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# MQTT Setup
latest_sensor_data = None
latest_prediction = None
latest_confidence = None

def scale_features(raw_data):
    """
    Scale features using min-max normalization.
    """
    scaled_data = (raw_data - FEATURE_MIN) / (FEATURE_MAX - FEATURE_MIN)
    logging.debug(f"Scaled features: {scaled_data}")
    return scaled_data

def predict_plant_health(scaled_data):
    """
    Perform inference using TensorFlow Lite model.
    """
    input_data = np.array([scaled_data], dtype=np.float32)
    logging.debug(f"Input data for model: {input_data}")

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    logging.debug(f"Raw model output: {output_data}")

    predicted_class = np.argmax(output_data)
    confidence = output_data[0][predicted_class]
    return CLASS_MAPPING[predicted_class], float(confidence)

def on_message(client, userdata, message):
    global latest_sensor_data, latest_prediction, latest_confidence
    if message.topic == DATA_TOPIC:
        payload = message.payload.decode('utf-8')
        try:
            latest_sensor_data = eval(payload)  # Convert to dict (ensure sender sends valid JSON)
            logging.info(f"Received data: {latest_sensor_data}")

            # Automatically process data and make predictions
            features = np.array([
                latest_sensor_data["soilMoisture"],
                latest_sensor_data["ambientTemp"],
                latest_sensor_data["soilTemp"],
                latest_sensor_data["humidity"],
                latest_sensor_data["lightIntensity"]
            ])
            scaled_features = scale_features(features)

            latest_prediction, latest_confidence = predict_plant_health(scaled_features)
            logging.info(f"Prediction: {latest_prediction}, Confidence: {latest_confidence}")

            # Publish prediction and confidence to MQTT
            prediction_message = {
                "prediction": latest_prediction,
                "confidence": latest_confidence
            }
            client.publish(PREDICTION_TOPIC, str(prediction_message))
            logging.info(f"Published prediction: {prediction_message}")

        except Exception as e:
            logging.error(f"Error processing message: {e}")


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logging.info("Connected to MQTT broker")
        client.subscribe(DATA_TOPIC)
    else:
        logging.error(f"Failed to connect to MQTT broker, return code: {rc}")

# Initialize MQTT client
client = mqtt.Client()
client.on_message = on_message
client.on_connect = on_connect

def start_mqtt():
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        logging.info("Started MQTT client")
    except Exception as e:
        logging.error(f"Error starting MQTT client: {e}")

@app.route('/')
def dashboard():
    return render_template("dashboard.html")

@app.route('/api/sensor_data')
def api_sensor_data():
    """
    API to get the latest sensor data and prediction.
    """
    if latest_sensor_data and latest_prediction is not None:
        return jsonify({
            "sensor_data": latest_sensor_data,
            "prediction": latest_prediction,
            "confidence": latest_confidence
        })
    return jsonify({"error": "No data available"}), 404

if __name__ == '__main__':
    start_mqtt()
    app.run(host='0.0.0.0', port=5000)
