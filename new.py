import serial
import time
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image, ImageOps

# Load the model
model = load_model("keras_model_plant.h5", compile=False)
class_names = open("labels_plant.txt", "r").readlines()

# Open Serial Port (Update COM port accordingly)
ser = serial.Serial('COM7', 115200, timeout=1)  # Change 'COM5' to your ESP8266 port

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Send initial forward command
ser.write(b'F\n')
print("Rover moving forward...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        continue

    # Preprocess image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    if index == 0:  # Weed Detected
        print(f"Weed DETECTED! Confidence: {confidence_score:.2f}")
        ser.write(b'S\n')  # Stop Rover
        time.sleep(1)  # Small delay to ensure it stops

        ser.write(b'A\n')  # Activate Actuator
        print("Actuator ON for 15 seconds")
        time.sleep(15)  # Keep actuator ON

        ser.write(b'X\n')  # Turn OFF actuator
        print("Actuator OFF, Resuming Forward Movement")
        ser.write(b'F\n')  # Resume Moving Forward
    else:
        print(f"No Weed Detected. Confidence: {confidence_score:.2f}")

    # Display frame
    cv2.putText(frame, f"{class_name}: {confidence_score:.2f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Weed Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
ser.close()

