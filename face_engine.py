import face_recognition
import numpy as np
import pickle
import os
from PIL import Image
from datetime import datetime

ENCODINGS_FILE = "models/encodings.pkl"

# Temporary memory for new faces
temporary_unknowns = {}


def recognize_face(image_file):
    global temporary_unknowns

    try:
        image_file.seek(0)
        pil_image = Image.open(image_file).convert("RGB")
        image = np.array(pil_image)

        if image is None or not isinstance(image, np.ndarray):
            return {"status": "invalid_image"}

    except Exception as e:
        return {"status": "invalid_image_exception", "error": str(e)}

    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        return {"status": "no_face"}

    if not os.path.exists(ENCODINGS_FILE):
        return {"status": "no_registered_faces"}

    with open(ENCODINGS_FILE, "rb") as f:
        known_data = pickle.load(f)

    if not known_data:
        return {"status": "no_registered_faces"}

    known_encodings = [person["encoding"] for person in known_data]
    face_encodings = face_recognition.face_encodings(image, face_locations)

    faces_output = []

    for i, (top, right, bottom, left) in enumerate(face_locations):

        padding = 60
        expanded_top = max(0, top - padding)
        expanded_left = max(0, left - padding)
        expanded_bottom = min(image.shape[0], bottom + padding)
        expanded_right = min(image.shape[1], right + padding)

        face_encoding = face_encodings[i]

        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(distances)
        distance = distances[best_match_index]
        confidence = round((1 - distance) * 100, 2)

        # 🔴 UNKNOWN FACE
        if confidence < 60:

            unknown_id = f"unknown_{i}"
            temporary_unknowns[unknown_id] = face_encoding

            faces_output.append({
                "name": "Unknown",
                "relationship": None,
                "confidence": confidence,
                "is_patient": False,
                "is_new_candidate": True,
                "unknown_id": unknown_id,
                "box": {
                    "top": expanded_top,
                    "right": expanded_right,
                    "bottom": expanded_bottom,
                    "left": expanded_left
                }
            })
            continue

        # ✅ RECOGNIZED FACE
        person = known_data[best_match_index]

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        person["last_seen"] = current_time

        history = person.get("history", [])
        latest_memory = history[-1] if history else None

        faces_output.append({
            "name": person["name"],
            "relationship": person.get("relationship"),
            "confidence": confidence,
            "is_patient": person.get("is_patient", False),
            "last_seen": current_time,
            "last_topic": latest_memory["summary"] if latest_memory else None,
            "last_date": latest_memory["date"] if latest_memory else None,
            "conversation_count": len(history),
            "recent_history": history[-5:],  # last 5 conversations
            "memory_insight": person.get("memory_insight"),
            "memory_confidence": person.get("memory_confidence", 0),
            "top_keywords": person.get("top_keywords", []),
            "is_new_candidate": False,
            "box": {
                "top": expanded_top,
                "right": expanded_right,
                "bottom": expanded_bottom,
                "left": expanded_left
            }
        })

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(known_data, f)

    return {
        "status": "faces_detected",
        "faces": faces_output
    }


# 🔵 REGISTER NEW FACE
def register_new_face(unknown_id, name, relationship):
    global temporary_unknowns

    if unknown_id not in temporary_unknowns:
        return {"status": "error", "message": "Unknown face expired"}

    new_encoding = temporary_unknowns.pop(unknown_id)

    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
    else:
        data = []

    data.append({
        "name": name,
        "relationship": relationship,
        "encoding": new_encoding,
        "is_patient": False,
        "last_seen": None,
        "history": [],
        "last_topic": None,
        "last_date": None,
        "memory_insight": None,
        "memory_confidence": 0,
        "top_keywords": []
    })

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    return {"status": "registered"}