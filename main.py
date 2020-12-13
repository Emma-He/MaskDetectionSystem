from enum import Enum
import cv2
import tensorflow as tf
import os
import time

class STATE(Enum):
    MASK = 0
    NO_MASK = 1
    NO_ONE = 2

# Prepares image for model specification
def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

def isMasked(file):
    # Manipulate image for faster recognition
    image = cv2.imread(file)
    cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detects faces in image
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    # print("[INFO] Found {0} Faces.".format(len(faces)))

    # Crops detected face (if applicable)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        # print("[INFO] Object found. Saving locally.")
        cv2.imwrite("cropface.jpg", roi_color)

    # Make prediction if there is a face
    if len(faces) != 0:
        print("[INFO] Image cropface.jpg written to filesystem")
        prediction = model.predict([prepare("cropface.jpg")])
        return STATE.NO_MASK if int(prediction[0][0]) == 1 else STATE.MASK

    return STATE.NO_ONE

model = tf.keras.models.load_model("model2")

fileName = "./webcam/" + str(time.time()) + ".jpg"
os.system("fswebcam -r 1280x960 " + fileName)
print(repr(isMasked(fileName)))