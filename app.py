from flask import Flask, request, jsonify, render_template
from face_engine import recognize_face, register_new_face
import pickle
from datetime import datetime
import re
from collections import Counter

app = Flask(__name__)

ENCODINGS_FILE = "models/encodings.pkl"

# ---------------------------------------------------
# 🔵 STOPWORDS (Lightweight NLP)
# ---------------------------------------------------

STOPWORDS = {
    "the","is","and","a","an","to","in","of","for","on","with",
    "we","i","you","it","that","this","about","were","was","are",
    "am","be","been","have","has","had","do","did","will","would",
    "my","our","your","they","them","their","but","or"
}

# ---------------------------------------------------
# 🔵 KEYWORD + SUMMARY GENERATOR
# ---------------------------------------------------

def generate_summary_and_keywords(text):

    text_clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    words = text_clean.split()

    meaningful = [w for w in words if w not in STOPWORDS and len(w) > 3]

    freq = Counter(meaningful)

    # Top 5 keywords
    keywords = [word for word, _ in freq.most_common(5)]

    # Short summary (first 6 meaningful words)
    summary_words = meaningful[:6]

    if not summary_words:
        summary = "Had a general conversation."
    else:
        summary = "Discussed " + " ".join(summary_words[:4]) + "."

    return summary, keywords


# ---------------------------------------------------
# 🔵 MEMORY INTELLIGENCE LAYER
# ---------------------------------------------------

def generate_memory_insight(history):

    if len(history) < 2:
        return None

    # Combine last 3 summaries
    ''
    recent = history[-3:]
    combined_text = " ".join([h["summary"] for h in recent])

    words = re.findall(r'\b[a-zA-Z]{4,}\b', combined_text.lower())
    meaningful = [w for w in words if w not in STOPWORDS]

    freq = Counter(meaningful)
    top = [word for word, _ in freq.most_common(3)]

    if not top:
        return None

    return f"You frequently discuss {', '.join(top)}."


def calculate_memory_confidence(history):
    if not history:
        return 0
    return min(100, 50 + len(history) * 10)


def extract_top_keywords_from_history(history, limit=5):

    all_keywords = []
    for h in history[-5:]:
        all_keywords.extend(h.get("keywords", []))

    freq = Counter(all_keywords)
    return [word for word, _ in freq.most_common(limit)]


# ---------------------------------------------------
# 🔵 ROUTES
# ---------------------------------------------------
@app.route('/')
def splash():
    return render_template("splash.html")
@app.route('/')

@app.route('/home')
def home():
    return render_template("index.html")

# ---------------------------------------------------
# 🔵 Face Recognition
# ---------------------------------------------------
@app.route('/calendar_data')
def calendar_data():

    try:
        with open(ENCODINGS_FILE, "rb") as f:
            people = pickle.load(f)

        calendar_map = {}

        for person in people:

            name = person.get("name")
            relationship = person.get("relationship")
            history = person.get("history", [])

            for entry in history:

                date_only = entry["date"].split(" ")[0]

                if date_only not in calendar_map:
                    calendar_map[date_only] = []

                calendar_map[date_only].append({
                    "name": name,
                    "relationship": relationship,
                    "summary": entry.get("summary"),
                    "raw": entry.get("raw"),
                    "keywords": entry.get("keywords", [])
                })

        return jsonify(calendar_map)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/calendar')
def calendar_page():
    return render_template("calendar.html")
@app.route('/recognize', methods=['POST'])
def recognize():
    image = request.files['image']
    result = recognize_face(image)
    return jsonify(result)


# ---------------------------------------------------
# 🔵 Save Conversation Note (INTELLIGENT VERSION)
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

        summary, keywords = generate_summary_and_keywords(note)

        for person in people:
            if person["name"] == name:

                if "history" not in person:
                    person["history"] = []

                # Append structured conversation
                person["history"].append({
                    "date": current_time,
                    "raw": note,
                    "summary": summary,
                    "keywords": keywords
                })

                history = person["history"]

                # Update metadata
                person["last_topic"] = summary
                person["last_date"] = current_time
                person["conversation_count"] = len(history)

                # 🔥 Intelligence Layer
                person["memory_insight"] = generate_memory_insight(history)
                person["memory_confidence"] = calculate_memory_confidence(history)
                person["top_keywords"] = extract_top_keywords_from_history(history)

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

@app.route('/reset-history', methods=['GET'])
def reset_history():
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            people = pickle.load(f)

        for person in people:
            person["history"] = []
            person["last_topic"] = None
            person["last_date"] = None
            person["conversation_count"] = 0
            person["memory_insight"] = None
            person["memory_confidence"] = 0
            person["top_keywords"] = []

        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(people, f)

        return "All conversations cleared (faces remain)"

    except Exception as e:
        return str(e), 500


# ---------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)