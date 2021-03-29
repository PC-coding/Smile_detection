import cv2

# face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# smile classifier
smile_classifier = cv2.CascadeClassifier('haarcascade_smile.xml')
# eye classifier
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

# webcam feed
webcam = cv2.VideoCapture(0)

while True:
    # reading webcam feed
    successful_frame, frame = webcam.read()

    # abort if error
    if not successful_frame:
        break

    # grayscale conversion, optomizes speed 
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # face detection
    face_detection = face_classifier.detectMultiScale(grayscale_frame)
    
    # smile detection
    smile_detection = smile_classifier.detectMultiScale(grayscale_frame, 
                                                        scaleFactor=1.7, 
                                                        minNeighbors=20)
    
    # eye detection
    eye_detection = eye_classifier.detectMultiScale(grayscale_frame,
                                                    scaleFactor=1.11,
                                                    minNeighbors=10)
    
    # run face_detection within each of the detected faces
    for (x, y, w, h) in face_detection:
        # draw a rectangle around the identified faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)
        
        # get the sub frame (use numpy N-dimensional array slicing)
        detected_faces = frame[y:y+h, x:x+w]
        
        # grayscale conversion
        gray_smile = cv2.cvtColor(detected_faces, cv2.COLOR_BGR2GRAY)
        
        # detect smiles
        detected_smile = smile_detection.detectMultiScale(gray_smile,
                                                        scalefactor=1.7,
                                                        minNeighbors=20)
        # find all smiles in the detected faces
        for (a, b, c, d) in detected_smile:
            # draw a rectangle around the smile
            cv2.rectangle(detected_faces, (a, b), (a + c + b + d), (50, 50, 200))
        
        # run eye detection
        for (a, b, c, d) in eye_detection:
            # draw a rectangle around the identified eyes
            cv2.rectangle(detected_faces, (a, b), (a+b, c+d), (255, 255, 255), 4)

    # labeling this face as smiling
    # will show the word "smiling" underneath the detected smile 
    if len(detected_smile) > 0:
        cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3,
        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    # show image in python program
    cv2.imshow('Smile Detection App', frame)

    # display, wait until you press a key to close program
    cv2.waitKey

# cleanup -> release webcam use, close python windows
webcam.release()
cv2.destroyAllWindows() 

print('successful')
    # # run smile_detection within each of the detected faces
    # for (x, y, w, h) in smile_detection:
    #     # draw a rectangle around the identified faces
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 200), 4)