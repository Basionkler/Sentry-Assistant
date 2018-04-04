import cv2
import glob
import os
import datasetRead
import datasetList
import shutil

from face_detection import find_faces

def remove_face_data(names):
    print("Removing previous processed faces...")
    shutil.rmtree("prepared_faces_dataset/")
    print("Done!")

def extract_faces(names):
    print("Extracting faces...")
    for name in names:
        print("Processing %s data..." % name)
        images = glob.glob('dataset/%s/*.jpg' % name)
        for file_number, image in enumerate(images):
            frame = cv2.imread(image)
            faces = find_faces(frame)
            for face in faces:
                path = ('prepared_faces_dataset/%s/' % name) 
                if not os.path.exists(path):
                    print("Creating data folder: %s" % path)
                    os.makedirs(path)
                filepath = path + '%s.jpg' % (file_number + 1)
                try:
                    cv2.imwrite(filepath, face[0])
                except:
                    print("Error in processing %s" % image)
    print("Face extraction finished")

def get_folder_list() :
    listOfDir = []
    for root, dirs, files in os.walk("dataset/", topdown=False):
       for name in dirs:
          listOfDir.append(name)
    listOfDir.sort()
    return listOfDir

if __name__ == '__main__':
    names = get_folder_list()
    remove_face_data(names)
    extract_faces(names)
