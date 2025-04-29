import cv2
import numpy as np
import time
import serial
import sqlite3
from keras.models import load_model  
from PIL import Image, ImageOps  

# Initialize Serial Connection
esp = serial.Serial("COM7", 115200, timeout=1)

# Load Model
model = load_model("keras_model_final.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Initialize Camera
cap = cv2.VideoCapture(0)

# Database Setup
def setup_database():
    conn = sqlite3.connect("weed_detection.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_detection():
    conn = sqlite3.connect("weed_detection.db")
    c = conn.cursor()
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO detections (timestamp) VALUES (?)", (timestamp,))
    conn.commit()
    conn.close()

def start_detection():
    esp.write(b'F')  # Start moving forward
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        confidence_score = prediction[0][index]
        class_name = class_names[index].strip()

        if index == 0:  # Weed detected
            esp.write(b'S')  # Stop Rover
            log_detection()
            time.sleep(1)
            esp.write(b'A')  # Activate actuator
            time.sleep(15)
            esp.write(b'X')  # Deactivate actuator
            time.sleep(1)
            esp.write(b'F')  # Resume movement
        else:
            print(f"No Weed Detected. Confidence: {confidence_score:.2f}")
            esp.write(b'F')
        
        cv2.putText(frame, f"{class_name}: {confidence_score:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Weed Detection", frame)

if __name__ == "__main__":
    setup_database()
    try:
        start_detection()
    finally:
        cap.release()
        cv2.destroyAllWindows()
