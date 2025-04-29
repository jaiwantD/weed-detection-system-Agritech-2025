# from keras.models import load_model  
# from PIL import Image, ImageOps  
# import numpy as np
# import cv2
# import serial
# import time
# import sqlite3

# # Load the trained model
# model = load_model("keras_model_final.h5", compile=False)

# # Load class labels
# class_names = open("labels.txt", "r").readlines()

# # Initialize camera
# cap = cv2.VideoCapture(0)  
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# # Establish Serial Connection to ESP8266
# esp = serial.Serial("COM5", 115200, timeout=1)  # Change COM7 to your correct port

# # Send initial stop command
# esp.write(b'S')  # Stop the rover
# esp.write(b'X')  # Ensure actuator is off
# print("Rover is stopped. Type 'start' to begin detection.")

# # Setup database
# conn = sqlite3.connect("weed_detection.db")
# c = conn.cursor()
# c.execute("""
#     CREATE TABLE IF NOT EXISTS detection (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         timestamp TEXT
#     )
# """)
# conn.commit()

# def log_weed_detection():
#     c.execute("INSERT INTO detection (timestamp) VALUES (datetime('now'))")
#     conn.commit()

# # Wait for user input to start
# while True:
#     user_input = input("Enter 'start' to begin: ").strip().lower()
#     if user_input == "start":
#         print("Starting weed detection...")
#         esp.write(b'F')  # Move forward **only after** start is typed
#         break  # Exit waiting loop

# # State variables
# weed_detected = False
# start_time = None
# waiting = False

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture image")
#         continue

#     # Convert frame to PIL image
#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     # Resize and preprocess image
#     image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
#     image_array = np.asarray(image)
#     normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
#     data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#     data[0] = normalized_image_array

#     # Predict the class
#     prediction = model.predict(data)
#     index = np.argmax(prediction)
#     class_name = class_names[index].strip()
#     confidence_score = prediction[0][index]
#     current_time = time.time()

#     if index == 1:  # Weed detected
#         if not weed_detected:
#             print(f"Weed DETECTED! Confidence: {confidence_score:.2f}")
#             esp.write(b'S')  # Stop Rover
#             start_time = current_time  # Record start time
#             weed_detected = True
#             waiting = True
#             esp.write(b'A')  # Activate actuator
#             log_weed_detection()  # Log weed detection in database
#         elif waiting and current_time - start_time >= 15:  # After 15 sec
#             esp.write(b'X')  # Deactivate actuator
#             time.sleep(1)  # Small pause
#             esp.write(b'F')  # Resume movement
#             waiting = False  # Reset waiting flag
#             weed_detected = False  # Reset detection flag
#     else:
#         print(f"No Weed Detected. Confidence: {confidence_score:.2f}")
#         esp.write(b'F')  # Keep moving forward

#     # Show camera feed
#     cv2.putText(frame, f"{class_name}: {confidence_score:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.imshow("Weed Detection", frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()
# esp.close()
# conn.close()


from keras.models import load_model  
from PIL import Image, ImageOps  
import numpy as np
import cv2
import serial
import time
import sqlite3

# Load the trained model
model = load_model("keras_model_final.h5", compile=False)

# Load class labels
class_names = open("labels.txt", "r").readlines()

# Initialize camera
cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Establish Serial Connection to ESP8266
esp = serial.Serial("COM5", 115200, timeout=1)  # Change COM5 to your correct port

# Send initial stop command
esp.write(b'S')  # Stop the rover
esp.write(b'X')  # Ensure actuator is off
print("Rover is stopped. Type 'start' to begin detection.")

# Setup database
conn = sqlite3.connect("weed_detection.db")
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS detection (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT
    )
""")
conn.commit()

def log_weed_detection():
    c.execute("INSERT INTO detection (timestamp) VALUES (datetime('now'))")
    conn.commit()

# Wait for user input to start
while True:
    user_input = input("Enter 'start' to begin: ").strip().lower()
    if user_input == "start":
        print("Starting weed detection...")
        esp.write(b'F')  # Move forward **only after** start is typed
        break  # Exit waiting loop

# State variables
weed_detected = False
start_time = None
waiting = False

while True:
    current_time = time.time()
    
    # Skip capturing images for 16 seconds after detecting weed
    if weed_detected and current_time - start_time < 16:
        continue
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        continue

    # Convert frame to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Resize and preprocess image
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict the class
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    current_time = time.time()

    if index == 1 and  confidence_score>=0.85:  # Weed detected
        print(f"Weed DETECTED! Confidence: {confidence_score:.2f}")
        esp.write(b'S')  # Stop Rover
        weed_detected = True
        start_time = current_time  # Record start time
        esp.write(b'A')  # Activate actuator
        log_weed_detection()  # Log weed detection in database
        time.sleep(15)  # Wait for 15 seconds before deactivating actuator
        esp.write(b'X')  # Deactivate actuator
        time.sleep(1)  # Small pause
        esp.write(b'F')  # Resume movement
        weed_detected = False  # Reset detection flag
    else:
        print(f"No Weed Detected. Confidence: {confidence_score:.2f}")
        esp.write(b'F')  # Keep moving forward

    # Show camera feed
    cv2.putText(frame, f"{class_name}: {confidence_score:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Weed Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
esp.close()
conn.close()
