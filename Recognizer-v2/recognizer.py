import face_recognition
import cv2
import datasetRead
import datasetList
import numpy as np

# This demo was made by Federico Ferri and Marco Pietrangeli
# The video is resized to the 25% than the original to improve performances
# In this version we still don't have the performances we desired
# it uses a single neural network (instead of two) to analyse both identities and emotions

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

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Rendering
font = cv2.FONT_HERSHEY_DUPLEX

#Emo Recognition
delay = 0
init = True

#Feature List
facial_features = [
	'chin',
	'left_eyebrow',
	'right_eyebrow',
	'nose_bridge',
	'nose_tip',
	'left_eye',
	'right_eye',
	'top_lip',
	'bottom_lip'
]


while True:
	# Grab a single frame of video
	ret, frame = video_capture.read()

	# Resize frame of video to 1/4 size for faster face recognition processing
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgb_small_frame = small_frame[:, :, ::-1]

	# Only process every other frame of video to save time
	if init or delay == 0:
		# Find all the faces and face encodings in the current frame of video
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
		face_landmarks_list = face_recognition.face_landmarks(frame)

		face_names = []

		for face_encoding in face_encodings:
			# See if the face is a match for the known face(s)
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

			name = "Unknown"

			# If a match was found in known_face_encodings, just use the first one.
			if True in matches:
				first_match_index = matches.index(True)
				name = known_face_names[first_match_index]

			face_names.append(name)

	# Display the results
	for (top, right, bottom, left), name in zip(face_locations, face_names):
		# Scale back up face locations since the frame we detected in was scaled to 1/2 size
		top *= 4
		right *= 4
		bottom *= 4
		left *= 4
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


	delay += 1
	delay %= 20

	for face_landmarks in face_landmarks_list:
		for facial_feature in facial_features:
			pts = np.array(face_landmarks[facial_feature], np.int32)
			pts = pts.reshape((-1,1,2))
			cv2.polylines(frame,[pts],True,(0,255,255))

	# Display the resulting image
	cv2.imshow('Video', frame)

	# Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	# Hit 'q' on the keyboard to quit!
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