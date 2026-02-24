import face_recognition
import numpy as np
import cv2
import time

last_person_id = None
last_recognition_time = 0
COOLDOWN_SECONDS = 60

def recognize_face(image_file):
    global last_person_id, last_recognition_time

    image = face_recognition.load_image_file(image_file)

    small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

    face_locations = face_recognition.face_locations(small_image)

    if not face_locations:
        return {"status": "no_face"}

    face_encodings = face_recognition.face_encodings(small_image, face_locations)

    # Temporary response
    return {
        "status": "recognized",
        "name": "Test User",
        "relationship": "Friend",
        "confidence": 90,
        "audio_message": "This is Test User, your friend."
    }