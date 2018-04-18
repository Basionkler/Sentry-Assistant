import cv2
import glob
import os

from face_detection import find_faces

def remove_face_data(emotions):
    print("Removing previous processed faces...")
    for emotion in emotions:
        filelist = glob.glob("emotion_dataset/prepared/%s/*" % emotion)
        for file in filelist:
            os.remove(file)
            
    print("Done!")

def extract_faces(emotions):
    print("Extracting faces...")
    for emotion in emotions:
        print("Processing %s data..." % emotion)
        images = glob.glob('emotion_dataset/raw/%s/*.jpg' % emotion)
        for file_number, image in enumerate(images):
            frame = cv2.imread(image)
            faces = find_faces(frame)
            for face in faces:
                try:
                    path = ('emotion_dataset/prepared/%s/%s.jpg' % (emotion, file_number + 1))
                    cv2.imwrite(path, face[0])
                except:
                    print("Error in processing %s" % image)

    print("Face extraction finished")


if __name__ == '__main__':
    emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"]
    remove_face_data(emotions)
    extract_faces(emotions)
