import face_recognition
import pickle
import os

ENCODINGS_FILE = "models/encodings.pkl"

def register_person(image_path, name, relationship):
    # Load image
    image = face_recognition.load_image_file(image_path)

    # Extract face encodings
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        print("No face found in image.")
        return

    encoding = encodings[0]

    # Load existing data
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
    else:
        data = []

    # Append new person
    data.append({
        "name": name,
        "relationship": relationship,
        "encoding": encoding
    })

    # Save back
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)

    print(f"{name} registered successfully.")

# Example usage
if __name__ == "__main__":
    register_person("test.jpg", "Sarthak", "Nephew")