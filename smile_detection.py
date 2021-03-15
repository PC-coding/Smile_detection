import cv2

# face classifier
face_classifier = cv2.CascadeClassifier('haarscascade_frontalface_default.xml')

# webcam feed
webcam = cv2.VideoCapture(0)

# reading webcam feed
successful_frame, frame = webcam.read()

# show image in python program
cv2.imshow('Smile Detection App', frame)

# display, wait until you press a key to close program
cv2.waitKey

print('successful')