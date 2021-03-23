import cv2

# face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_classifier = cv2.CascadeClassifier('haarcascade_smile.xml')

# webcam feed
webcam = cv2.VideoCapture(0)

while True:
    # reading webcam feed
    successful_frame, frame = webcam.read()

    # abort if error
    if not successful_frame:
        break

    # grayscale conversion
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # face detection
    face_detection = face_classifier.detectMultiScale(grayscale_frame)
    
    # smile detection
    smile_detection = smile_classifier.detectMultiScale(grayscale_frame, 
                                                        scaleFactor=1.7, 
                                                        minNeighbors=20)

    # run face_detection within each of the detected faces
    for (x, y, w, h) in face_detection:
        # draw a rectangle around the identified faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)
        
        # get the subframe
        detected_faces = (x, y, w, h)
        
        # grayscale conversion
        gray_smile = cv2.cvtColor(detected_faces, cv2.COLOR_BGR2GRAY)
        
        # detect smiles
        detected_smile = smile_detection.detectMultiScale(gray_smile,
                                                        scalefactor=1.7,
                                                        minNeighbors=20)
        # find all smiles in the detected faces
        for (a, b, c, d) in detected_smile:

    # run smile_detection within each of the detected faces
    for (x, y, w, h) in smile_detection:
        # draw a rectangle around the identified faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 200), 4)

    # show image in python program
    cv2.imshow('Smile Detection App', frame)

    # display, wait until you press a key to close program
    cv2.waitKey

# cleanup -> release webcam use, close python windows
webcam.release()
cv2.destroyAllWindows() 

print('successful')