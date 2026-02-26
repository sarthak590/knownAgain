import face_recognition
import pickle
import os

ENCODINGS_FILE = "models/encodings.pkl"

def register_person(image_path, name, relationship, is_patient=False):

    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        print("No face found in image.")
        return

    encoding = encodings[0]

    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
    else:
        data = []

    data.append({
        "name": name,
        "relationship": relationship,
        "encoding": encoding,
        "is_patient": is_patient,
        "last_seen": None,
        "last_topic": None
    })

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"{name} registered successfully.")


if __name__ == "__main__":
    # Make test.jpg the PATIENT
    register_person("test.jpg", "Patient", "Self", is_patient=True)