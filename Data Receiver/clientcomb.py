import serial
import requests
import datetime

arduino = serial.Serial('COM3', 9600)  # adjust COM port

while True:
    try:
        line = arduino.readline().decode().strip()
        temp, motion, light, rfid = line.split(",")

        now = datetime.datetime.now()
        data = {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H"),  # only hour
            "temperature": float(temp),
            "motion": int(motion),
            "light": int(light),
            "rfid": int(rfid),
        }

        requests.post("http://192.168.20.20:5000/data", json=data)
        print(f"Sent: {data}")

    except Exception as e:
        print(f"Error: {e}")
