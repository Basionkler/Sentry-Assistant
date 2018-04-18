import cv2
import glob
import os

from face_detection import find_faces

def remove_face_data():
    print("Removing previous processed faces...")
    for gender in genders:
        filelist = glob.glob("gender_dataset/prepared_gender/%s/*" % gender)
    for file in filelist:
        os.remove(file)

    print("Done!")

# def extract_faces():
#     print("Extracting faces")
#     for i in range (0, 10):
#         images = glob.glob('../data/imdb_crop/%s/*.jpg' % i)

#         for file_number, image in enumerate(images):
#             frame = cv2.imread(image)
#             faces = find_faces(frame)

#             for face in faces:
#                 try:
#                     cv2.imwrite("../data/gender/assorted/%s/%s.jpg" % (i, (file_number + 1)), face[0])
#                 except:
#                     print("Error in processing %s" % image)

def extract_faces(genders):
    print("Extracting faces")
    for gender in genders:
        images = glob.glob('gender_dataset/raw_gender/%s/training_set/*.jpg' % gender)
        for file_number, image in enumerate(images):
            frame = cv2.imread(image)
            faces = find_faces(frame)

            for face in faces:
                try:
                    cv2.imwrite("gender_dataset/prepared_gender/%s/%s.jpg" % (gender, (file_number + 1)), face[0])

                except:
                    print("Error in processing %s" % image)

    print("Face extraction finished")

if __name__ == '__main__':
    genders = ["female", "male"]
    remove_face_data()
    extract_faces(genders)
