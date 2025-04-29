import os
import subprocess
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__, template_folder="templates")

# Load crop prediction model and scalers
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/start-detection", methods=["GET"])
def start_detection():
    python_exec = os.path.join("D:\\weeddetection\\env", "Scripts", "python.exe")
    subprocess.Popen(["start", "cmd", "/k", python_exec, "noblockdb.py"], shell=True, cwd="D:\\weeddetection")
    return "Detection Started"

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    data = request.json
    feature_list = [data['N'], data['P'], data['K'], data['temp'], data['humidity'], data['ph'], data['rainfall']]
    single_pred = np.array(feature_list).reshape(1, -1)
    
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    
    prediction = model.predict(final_features)
    crop_name = crop_dict.get(prediction[0], "Unknown")
    
    return jsonify({"predicted_crop": crop_name})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
