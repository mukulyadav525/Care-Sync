from flask import Flask, request, jsonify
import os
import csv
import pytz
import sys  # Import sys to exit the program
from datetime import datetime

app = Flask(__name__)



BASE_DIR = "/home/megha21337/mukul/Care-Sync/Users"

# Ask user for directory name when the server starts
directory_name = input("Enter Username: ")
current_directory = os.path.join(BASE_DIR, directory_name)

# Check if the directory exists
if not os.path.exists(current_directory):
    print("Error: Username does not exist!")
    sys.exit(1)  # Exit with error code 1

print(f"Username exists. Proceeding with directory: {current_directory}")
# Initialize session state
current_file = None

# Set IST timezone
IST = pytz.timezone("Asia/Kolkata")


@app.route('/start-session', methods=['POST'])
def start_session():
    """
    Start a new session by creating a new CSV file inside the chosen directory.
    """
    global current_file
    try:
        if current_file is not None:
            return jsonify({"error": "A session is already active. End the session before starting a new one."}), 400

        # Create a new CSV file inside the pre-set directory
        timestamp = datetime.now(pytz.utc).astimezone(IST).strftime("%d-%m-%y_%H-%M-%S")
        current_file = os.path.join(current_directory, f"{timestamp}.csv")

        with open(current_file, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "Hour", "Minute", "Second", "Millisecond",
                "Red", "IR", "AccelX", "AccelY", "AccelZ",
                "GyroX", "GyroY", "GyroZ", "GSR"
            ])

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

        sensor_data_list = request.json  # Get JSON data

        if not isinstance(sensor_data_list, list):
            return jsonify({"error": "Invalid data format, expected a list."}), 400

        with open(current_file, "a", newline='') as file:
            writer = csv.writer(file)
            for sensor_data in sensor_data_list:
                if not isinstance(sensor_data, dict):
                    return jsonify({"error": "Invalid record format, expected a dictionary"}), 400
                writer.writerow([
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
                ])
        return jsonify({"message": "Batch data appended successfully"}), 200
    except Exception as e:
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

        current_file = None
        return jsonify({"message": "Session ended and file closed."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
