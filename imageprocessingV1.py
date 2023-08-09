import numpy as np
import cv2

frame = cv2.imread('assets/seniorpic1.jpg')
frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")


counter=0

#changing the image to grayscale and then initiating the face finding algorithm on it
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 10)

#doing the following for each face
for (x, y, w, h) in faces:
    #finding where to put the rectangle and drawing it there
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 4)
    roi_gray = gray[y:y+w, x:x+w]
    roi_color = frame[y:y+h, x:x+w]

    #saving each face into its own file
    face_filename = f"assets/faces/face_{counter}.jpg"
    cv2.imwrite(face_filename, roi_color)
    counter += 1

    img = cv2.imread('assets/summerpic.jpg',0)
    #img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)
    template = cv2.imread(face_filename,0)
    h, w = template.shape
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    for method in methods:
        img2 = img.copy()
        result = cv2.matchTemplate(img2, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            location = min_loc
            threshold = 0.2
        else:
            location = max_loc
            threshold = 0.75
        
        #check if the picture is a match based on its relation to the threshold
        if max_val >= threshold:
            print(f"Match found using method {method} on face {counter}")
        else:
            print(f"No match found using method {method} on face {counter}")

        bottom_right = (location[0] + w, location[1] + h)
        cv2.rectangle(img2, location, bottom_right, 0, 5)
        cv2.imshow('match', img2)




cv2.imshow('facial recognition program', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
