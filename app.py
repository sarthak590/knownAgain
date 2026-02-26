from flask import Flask, request, jsonify, render_template
from face_engine import recognize_face, register_new_face
import pickle
import re
from datetime import datetime

app = Flask(__name__)

ENCODINGS_FILE = "models/encodings.pkl"


# ---------------------------------------------------
# 🔵 Lightweight Keyword Summary Generator
# ---------------------------------------------------

def generate_summary(text):
    stopwords = {
        "i","me","my","we","our","you","your",
        "is","are","was","were","the","a","an",
        "and","or","but","to","of","in","on",
        "for","with","about","that","this","it",
        "have","had","has","will","would","could",
        "should","they","them","their"
    }

    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

    keywords = [w for w in words if w not in stopwords]

    unique_keywords = list(dict.fromkeys(keywords))

    top_keywords = unique_keywords[:3]

    if not top_keywords:
        return "Had a general conversation."

    return "Discussed " + ", ".join(top_keywords) + "."


# ---------------------------------------------------
# 🔵 Routes
# ---------------------------------------------------

@app.route('/')
def home():
    return render_template("index.html")


# ---------------------------------------------------
# 🔵 Face Recognition
# ---------------------------------------------------

@app.route('/recognize', methods=['POST'])
def recognize():
    image = request.files['image']
    result = recognize_face(image)
    return jsonify(result)


# ---------------------------------------------------
# 🔵 Save Conversation Note
# ---------------------------------------------------

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
        with open(ENCODINGS_FILE, "rb") as f:
            people = pickle.load(f)

        person_found = False
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = generate_summary(note)

        for person in people:
            if person["name"] == name:

                # Initialize history if not exists
                if "history" not in person:
                    person["history"] = []

                person["history"].append({
                    "date": current_time,
                    "transcript": note,
                    "summary": summary
                })

                # Update metadata
                person["conversation_count"] = len(person["history"])
                person["last_topic"] = summary
                person["last_date"] = current_time

                person_found = True
                break

        if not person_found:
            return jsonify({"status": "error", "message": "Person not found"}), 404

        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(people, f)

        return jsonify({"status": "note_saved"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------------------------------------
# 🔵 Register Unknown Face
# ---------------------------------------------------

@app.route('/register-new', methods=['POST'])
def register_new():
    data = request.get_json()

    if not data:
        return jsonify({"status": "error", "message": "No data received"}), 400

    unknown_id = data.get("unknown_id")
    name = data.get("name")
    relationship = data.get("relationship")

    if not unknown_id or not name or not relationship:
        return jsonify({"status": "error", "message": "Missing fields"}), 400

    result = register_new_face(unknown_id, name, relationship)
    return jsonify(result)


# ---------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)