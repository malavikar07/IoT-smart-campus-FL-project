from flask import Flask, request, jsonify
import csv
import os
from datetime import datetime

def is_now_between(start_datetime: datetime, end_datetime: datetime) -> bool:
    """
    Return True if the current local time is between start_time and end_time.

    Handles intervals that cross midnight, e.g. 23:00–07:00.
    """
    now = datetime.now()
    if start_datetime <= end_datetime:
        return start_datetime <= now <= end_datetime
    else:
        return now >= start_datetime or now <= end_datetime

app = Flask(__name__)
CSV_FILE = "sensor_log.csv"

# Write header if file doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "hour", "is_holiday", "maintenance_going_on", "light_sensor", "motion_sensor", "door_lock_sensor", "temperature_sensor","room"])


def get_last_row(path, has_header=True):
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            if has_header:
                header = next(reader, None)
            last = None
            for row in reader:
                last = row
        return last
    except Exception as e:
        print(e)

start_datetime,end_datetime=get_last_row('maintenance.csv')

start_datetime = datetime.strptime(start_datetime, "%Y-%m-%dT%H:%M:%S")
end_datetime = datetime.strptime(end_datetime, "%Y-%m-%dT%H:%M:%S") 

@app.route('/data', methods=['POST'])
def receive_data():
    data = request.json

    date_obj = datetime.strptime(data["date"], "%Y-%m-%d")
    holiday = 1 if date_obj.weekday() >= 5 else 0  # Saturday/Sunday
    maintenance = 1 if start_datetime<=datetime.now()<=end_datetime else 0  # You can update this based on any condition or file check

    row = [
        data["date"],
        data["time"],
        holiday,
        maintenance,
        data["light"],
        data["motion"],
        data["rfid"],
        data["temperature"],
        1#based on the room Classroom-0,Computer_lab-1,Hallway-2
    ]

    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return jsonify({"status": "success", "received": row})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
