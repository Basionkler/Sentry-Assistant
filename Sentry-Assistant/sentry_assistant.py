import cv2
import lbphrecognizer_train
from face_detection import find_faces
from lbphrecognizer_train import detect_face
from dict_create import cache_results
from collections import defaultdict

# This project was made by Federico Ferri and Marco Pietrangeli
# For Machine Learning and Sistemi Intelligenti Per Internet.
# The system uses a Neural Network to recognize people and their emotions
# This is a possible application to Analyze webcams of public places (like stations)
# To prevent and take control on High-Risk situations.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load model
faces = lbphrecognizer_train.get_folder_list()
faces.append("Unknown")
lbph_face_name = cv2.face.LBPHFaceRecognizer_create()
lbph_face_name.read('trainedmodel/lbph_trained_model.yml')

emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"]
fisher_face_emotion = cv2.face.FisherFaceRecognizer_create()
fisher_face_emotion.read('models/emotion_classifier_model.xml')

# Rendering
font = cv2.FONT_HERSHEY_DUPLEX
color = (255, 255, 255)

# Risk
threat = 0
threat_color = (255,255,255)
risk = ""
i = 0
fps = 0

# Caching
face_is_found = False
dictionaries = []
starting_energy = 100.0
energy = 0
energies = []
labels = []
scores = []

print("- Type 'q' to quit")
print("- Type 's' to save a screenshot")

while True:
    i = 0
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    for normalized_face, (x, y, w, h) in find_faces(small_frame):
        x *= 2
        y *= 2
        h *= 2
        w *= 2
        energy = 0

        face_prediction, confidence = lbph_face_name.predict(normalized_face)
        emotion_prediction = fisher_face_emotion.predict(normalized_face)
        if(len(energies)-1 < i):
            energies.insert(i, starting_energy)
            dictionaries.insert(i, defaultdict(list))
            labels.insert(i, 0)
            scores.insert(i, 0)
        label, score, energy = cache_results(face_prediction, confidence, dictionaries[i], energies[i], 0.015)
        labels[i] = label
        scores[i] = score
        energies[i] = energy
        if (score > 50):
            name = faces[labels[i]]
        else:
            name = faces[-1]
        emo = emotions[emotion_prediction[0]]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100, 255, 100), 1)
        if(energy > 0.1):
            cv2.putText(frame, "Searching", (x,y-10), font, 1, (255, 255, 255), 2)
        else: 
            cv2.putText(frame, name, (x,y-10), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, emo, (x, y+h+32), font, 1, (100, 255, 100), 2)

        # Draw Threat entity - must be improved
        if(name is not "Unknown"):
            threat += 0.50
        else:
            threat = 0

        # Check on Emotions
        if(emo == "neutral" or emo == "happy"):
            threat += 0.10
        elif(emo == "angry"):
            threat += 0.25
        else:
            threat += 0.15

        i += 1

    # Preparing Data for rendering
    if(threat < 0.15):
        risk = "No Risk"
        threat_color = (0, 255, 0) # Solid Green
    elif(threat >= 0.15 and threat < 0.25):
        risk = "Low Risk"
        threat_color = (47, 255, 173) # Green Yellow
    elif(threat >= 0.25 and threat < 0.5):
        risk = "Medium Risk"
        threat_color = (0, 255, 255) # Yellow
    elif(threat >= 0.5 and threat <= 0.6):
        risk = "Medium/High Risk"
        threat_color = (0, 128, 255) # Orange
    else:
        risk = "High Risk"
        threat_color = (0, 0, 255) # Red

    # Render threat circle - params (img, center, radius, color, thickness)
    cv2.circle(frame, (20, 20), 10, threat_color, -1)
    cv2.putText(frame, risk, (35, 30), font, 1, threat_color)
    threat = 0

    # Display the resulting image
    cv2.imshow('VIDEO', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('Screen.jpg', frame)
        print("Screenshot saved.")

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
