import face_recognition
import os
import cv2
import pickle

# this file is for training and pickling only

KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6  # default is 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"  # can also use 'hog' cnn runs slower on only cpu than hog

print("loading known faces:")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    print(name)
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)
    print("Finished training:", name)

print("processing unknown faces:")


# pickle the training face & name lists.
with open('pickle_files/known_face_pickle1.pkl', 'wb') as known_face_pickle:
    pickle.dump(known_faces, known_face_pickle)
with open('pickle_files/known_name_pickle1.pkl', 'wb') as known_name_pickle:
    pickle.dump(known_names, known_name_pickle)

