import numpy as np
import pickle

# Importing model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
    7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
    12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
    17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
    21: "Chickpea", 22: "Coffee"
}

# User input
N = float(input("Enter Nitrogen value: "))
P = float(input("Enter Phosphorus value: "))
K = float(input("Enter Potassium value: "))
temp = float(input("Enter Temperature value: "))
humidity = float(input("Enter Humidity value: "))
ph = float(input("Enter pH value: "))
rainfall = float(input("Enter Rainfall value: "))

# Prepare features
feature_list = [N, P, K, temp, humidity, ph, rainfall]
single_pred = np.array(feature_list).reshape(1, -1)

# Scale features
scaled_features = ms.transform(single_pred)
final_features = sc.transform(scaled_features)

# Predict crop
prediction = model.predict(final_features)

# Output result
crop = crop_dict.get(prediction[0], "Unknown")
print(f"The best crop to be cultivated is: {crop}")
