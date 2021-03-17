import cv2

# face classifier
face_classifier = cv2.CascadeClassifier('haarscascade_frontalface_default.xml')

# webcam feed
webcam = cv2.VideoCapture(0)

while True:
    # reading webcam feed
    successful_frame, frame = webcam.read()

    # abort if error
    if not successful_frame:
        break

    # show image in python program
    cv2.imshow('Smile Detection App', frame)

    # display, wait until you press a key to close program
    cv2.waitKey

# cleanup -> release webcam use, close python windows
webcam.release()
cv2.destroyAllWindows() 

print('successful')