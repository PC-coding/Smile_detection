import cv2

# face classifier
face_classifier = cv2.CascadeClassifier('haarscascade_frontalface_default.xml')

# webcam feed
webcam = cv2.VideoCapture(0)

successful_frame, frame = webcam.read()


print('successful')