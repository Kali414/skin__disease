from flask import Flask,request, jsonify,render_template

from flask_cors import CORS

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.models import load_model
import numpy as np
import cv2



from google import generativeai as genai


# Set up your API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

genmodel= genai.GenerativeModel(model_name="gemini-2.0-flash")



model=load_model('model.keras')

classes=np.load('classes.npy')

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    if(request.method == 'GET'):
        return jsonify({"message": "GET method not allowed"})

    image = request.files['image']
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (250, 250))
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    prediction = np.argmax(prediction)
    print(prediction)


    contents=f"Explain about the disease {classes[prediction]} in short detail and suggest some remedies ",

    response = genmodel.generate_content(contents)

    print(response.text)

    return jsonify({"prediction": classes[prediction],"response": response.text}),200


if __name__ == '__main__':
    app.run(debug=True)