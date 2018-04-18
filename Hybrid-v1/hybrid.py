import face_recognition
import cv2
import datasetRead
import datasetList
from face_detection import find_faces

# This demo was made by Federico Ferri and Marco Pietrangeli.
# In this initial version we use 2 different Neural Network.
# The first one get facial features to identify people identities
# The second one get facial features to identify people emotions
# The main reason we decided to rework this project is because
# Performances and Results were under our expectation

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load pictures and learn how to recognize.
if(input('ricaricare configurazione precedente? (y/n)\n') == 'y'):
    encodings, facesname = datasetRead.dataset_import_from_files()
else:
    datasetList.dataset_list_update()
    encodings, facesname = datasetRead.dataset_read()

# Create arrays of known face encodings and their names
known_face_encodings = encodings
known_face_names = facesname

#Modello Emo/Gender Detection
# Load model
fisher_face_emotion = cv2.face.FisherFaceRecognizer_create()
fisher_face_emotion.read('models/emotion_classifier_model.xml')
fisher_face_gender = cv2.face.FisherFaceRecognizer_create()
fisher_face_gender.read('models/gender_classifier_model.xml')
emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Rendering
font = cv2.FONT_HERSHEY_DUPLEX
color = (255, 255, 255)

#Emo Recognition
delay = 0
init = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # cv2.imwrite('Video.jpg', frame)

    # Only process every other frame of video to save time
    if init or delay == 0:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)

            name = "Unknown"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/2 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        #cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    #EMO/GENDER RECOGNITION
    for normalized_face, (x, y, w, h) in find_faces(frame):
        if init or delay == 0:
            init = False
            emotion_prediction = fisher_face_emotion.predict(normalized_face)
            gender_prediction = fisher_face_gender.predict(normalized_face)
        if (gender_prediction[0] == 0):
            color = (179,52,255)
        else:
            color = (255,0,0)        
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(frame, emotions[emotion_prediction[0]], (x,y-10), font, 1.5, color, 2)

    delay += 1
    delay %= 1

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Hit 'u' on the keyboard to update!
    if cv2.waitKey(2) & 0xFF == ord('u'):
        print("SYSTEM UPDATING...")
        encodings, facesname = datasetRead.dataset_update()
        for item in encodings:
            known_face_encodings.append(item)
        for item in facesname:
            known_face_names.append(item)
        print("SYSTEM UPDATED SUCCESSFULLY!")

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
