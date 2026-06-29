from flask import Flask, request, jsonify
import os
import csv
import pytz
from datetime import datetime
import json  # Ensure this is included at the top of your file

app = Flask(__name__)

# Directory to save CSV files 
UPLOAD_DIR = "/home/megha21337/HRV_DATA_FEB"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state
current_file = None

# Set IST timezone
IST = pytz.timezone("Asia/Kolkata")


@app.route('/start-session', methods=['POST'])
def start_session():
    """
    Start a new session and create a CSV file with headers, if no session is active.
    """
    global current_file
    print(current_file)
    try:
        if current_file is not None:
            print(f"Session already active. Current file: {current_file}")
            return jsonify({"error": "A session is already active. End the session before starting a new one."}), 400

        timestamp = datetime.now(pytz.utc).astimezone(IST).strftime("%d-%m-%y_%H-%M-%S")
        current_file = os.path.join(UPLOAD_DIR, f"{timestamp}.csv")

        with open(current_file, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Hour", "Minute", "Second", "Millisecond",
                "Red", "IR", "AccelX", "AccelY", "AccelZ",
                "GyroX", "GyroY", "GyroZ", "GSR"
            ])

        print(f"Session started. File created: {current_file}")
        return jsonify({"message": f"Session started. File: {current_file}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/append-data', methods=['POST'])
def append_data():
    """
    Append batch sensor data to the active session's CSV file.
    """
    global current_file
    try:
        if not current_file:
            return jsonify({"error": "No active session. Start a session first."}), 400

        # DEBUG: Print confirmation that data is received
        

        # Attempt to parse JSON
        try:
            sensor_data_list = request.json  # Simplified to use request.json directly
        except Exception:
            return jsonify({"error": "Invalid JSON format"}), 400

        # Check if parsed data is a list
        if not isinstance(sensor_data_list, list):
            return jsonify({"error": "Invalid data format, expected a list."}), 400

        # Append data to the CSV file
        with open(current_file, "a", newline='') as file:
            writer = csv.writer(file)

            for sensor_data in sensor_data_list:
                # Ensure each record is a dictionary
                if not isinstance(sensor_data, dict):
                    return jsonify({"error": "Invalid record format, expected a dictionary"}), 400

                # Extract fields with default empty string if key is missing
                row = [
                    sensor_data.get("Hour", ""),
                    sensor_data.get("Minute", ""),
                    sensor_data.get("Second", ""),
                    sensor_data.get("Millisecond", ""),
                    sensor_data.get("Red", ""),
                    sensor_data.get("IR", ""),
                    sensor_data.get("AccelX", ""),
                    sensor_data.get("AccelY", ""),
                    sensor_data.get("AccelZ", ""),
                    sensor_data.get("GyroX", ""),
                    sensor_data.get("GyroY", ""),
                    sensor_data.get("GyroZ", ""),
                    sensor_data.get("GSR", "")
                ]
                writer.writerow(row)

        # Print confirmation message for debugging
        
        return jsonify({"message": "Batch data appended successfully"}), 200

    except Exception as e:
        # Log the error to the console
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/end-session', methods=['POST'])
def end_session():
    """
    Ends the active session.
    """
    global current_file
    try:
        if not current_file:
            return jsonify({"error": "No active session to end."}), 400

        print(f"Session ended. File: {current_file}")
        current_file = None
        return jsonify({"message": "Session ended and file closed."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
