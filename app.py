from flask import Flask, request, jsonify, render_template
from face_engine import recognize_face, register_new_face
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")


# 🔵 Face Recognition
@app.route('/recognize', methods=['POST'])
def recognize():
    image = request.files['image']
    result = recognize_face(image)
    return jsonify(result)


# 🔵 Save Conversation Note
@app.route('/add-note', methods=['POST'])
def add_note():
    data = request.get_json()

    if not data:
        return jsonify({"status": "error", "message": "No data received"}), 400

    name = data.get("name")
    note = data.get("note")

    if not name or not note:
        return jsonify({"status": "error", "message": "Missing name or note"}), 400

    try:
        with open("models/encodings.pkl", "rb") as f:
            people = pickle.load(f)

        person_found = False

        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for person in people:
            if person["name"] == name:

                # Create conversations list if not exists
                if "conversations" not in person:
                    person["conversations"] = []

                person["conversations"].append({
                    "date": current_time,
                    "note": note
                })

                # Also keep last_topic for compatibility
                person["last_topic"] = note

                person_found = True
                break

        if not person_found:
            return jsonify({"status": "error", "message": "Person not found"}), 404

        with open("models/encodings.pkl", "wb") as f:
            pickle.dump(people, f)

        return jsonify({"status": "note_saved"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# 🔵 NEW ROUTE → Register Unknown Face
@app.route('/register-new', methods=['POST'])
def register_new():
    data = request.get_json()

    unknown_id = data.get("unknown_id")
    name = data.get("name")
    relationship = data.get("relationship")

    if not unknown_id or not name or not relationship:
        return jsonify({"status": "error", "message": "Missing fields"}), 400

    result = register_new_face(unknown_id, name, relationship)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)