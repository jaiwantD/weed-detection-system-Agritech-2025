from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

import cv2  # OpenCV for continuous image capture

# Load the model
model = load_model("keras_model_final.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Initialize camera
cap = cv2.VideoCapture(0)  # Change to 1 if using an external camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        continue

    # Convert frame to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Resize and preprocess image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict the class
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Print result
    if index==0 or index==2:
        print(f"NO Weed Detected Confidence: {confidence_score:.2f}")
    else:
        print(f"Weed DETECTED!. Confidence: {confidence_score:.2f}")

    # Display the frame
    cv2.putText(frame, f"{class_name}: {confidence_score:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Weed Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
