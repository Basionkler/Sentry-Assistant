import cv2
import glob
import numpy as np
import random
import os
import time
from face_detection import find_faces

# Create Recognizer
lbph_face_name = cv2.face.LBPHFaceRecognizer_create()
data_folder_path = 'prepared_faces_dataset/'
destination_file = 'trainedmodel/lbph_trained_model.yml'

#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #load OpenCV face detector, I am using LBP which is fast
    #there is also a more accurate but slow: Haar classifier
    #face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=35);

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]
    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]

#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
#of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
     
    #------STEP-1--------
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    dirs.sort()
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    #Analisys-parameters
    total_faces = 0
    label = 0
    #let's go through each directory and read images within it
    for dir_name in dirs:
         
        #build path of directory containing images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "" + dir_name
        print(subject_dir_path)
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        #------STEP-3--------
        #go through each image name, read image, 
        #detect face and add face to list of faces
        for image_name in subject_images_names:
            print(image_name)
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;
     
            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)
             
            #display an image window to show the image 
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
             
            #detect face
            for face, (x, y, w, h) in find_faces(image): 
                #------STEP-4--------d
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
        label += 1

        total_faces += 1
        print(labels)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
    
    lbph_face_name.train(faces, np.array(labels))
    return total_faces, len(labels)

# Get the array-list of faces
def get_folder_list() :
    listOfDir = []
    for root, dirs, files in os.walk(data_folder_path, topdown=False):
        for name in dirs:
            listOfDir.append(name.replace("_", " "))
    listOfDir.sort()
    return listOfDir

if __name__ == '__main__':
    if(os.path.exists(data_folder_path)):
        total_faces, total_labels = prepare_training_data(data_folder_path)
        print("Fetching data from %s" % data_folder_path)
        print("Found", total_faces, "faces and", total_labels, "labels")
        if not os.path.exists("trainedmodel/"):
        	print("Creating data folder: trainedmodel/")
        	os.makedirs("trainedmodel/")
        lbph_face_name.write(destination_file)
        print("Data written successfully!")
    else:
        print("%s does NOT exist" % data_folder_path)