from flask import Flask, request, jsonify
from face_engine import recognize_face

app = Flask(__name__)

@app.route('/')
def home():
    return "knownAgain AI Memory Assistant Running"

@app.route('/recognize', methods=['POST'])
def recognize():
    image = request.files['image']
    result = recognize_face(image)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)