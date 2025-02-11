import cv2
import numpy as np
import face_recognition

imgHari = face_recognition.load_image_file('ImagesBasic/saurav.jpg')
imgHari = cv2.cvtColor(imgHari,cv2.COLOR_BGR2RGB)
imgHarish = face_recognition.load_image_file('ImagesBasic/dhruv.jpg')
imgHarish = cv2.cvtColor(imgHarish,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgHari)[0]
encodeHari = face_recognition.face_encodings(imgHari)[0]
cv2.rectangle(imgHari, (faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocHarish = face_recognition.face_locations(imgHarish)[0]
encodeHarish = face_recognition.face_encodings(imgHarish)[0]
cv2.rectangle(imgHarish, (faceLocHarish[3],faceLocHarish[0]),(faceLocHarish[1],faceLocHarish[2]),(255,0,255),2)
# these are the 128 mesurements of both the faces.

results = face_recognition.compare_faces([encodeHari],encodeHarish)
faceDis = face_recognition.face_distance([encodeHari],encodeHarish)
print(results, faceDis)
cv2.putText(imgHarish,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255),2)

cv2.imshow('Hari', imgHari)
cv2.imshow('Harish', imgHarish)
cv2.waitKey(0)