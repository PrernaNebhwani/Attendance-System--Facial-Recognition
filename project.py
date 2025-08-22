import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from datetime import date

#Path has the directory where we are having our Dataset(images of Known people)
path = 'image for Attendance'

#Images contains the all the images from the folder of dataset
images = []

#classNames contains the name of particular images from dataset
classNames = []

#myList contains list of the path 
myList = os.listdir(path)

#it will print the all the files in the path 
print(myList)

#In this loop we are currently reading file from path and putting in the images variable
#After splitting the text of file it will insert in classNames
#for eg :- prajval.jpg is the file name so 
#prajval will insert in className
#image of prajval will insert in images

for file in myList:
    currentImage = cv2.imread(f'{path}/{file}')
    images.append(currentImage)
    classNames.append(os.path.splitext(file)[0])
print(classNames)


#function to find the encoding of the images..

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encodeImage = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeImage)
    return encodeList

#A global variable to keep the count of known people in the list

serialNo = 0

#A function to mark the attendance of the students in the list
#here we compare the current face in the list if it is already present it will not mark present 
#else it will present

def markAttendance(name):
    with open('record.csv','r+') as f:
        myDataList = f.readlines()
        namelist = []
        for line in myDataList:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist and name != 'Unknown Face':
            now = datetime.now()
            timeString  = now.strftime('%H::%M::%S')
            dates = date.today()
            dateToday = dates.strftime('%d-%m-%Y')
            global serialNo
            serialNo = serialNo + 1
            f.writelines(f'\n{name},{serialNo},{timeString},{dateToday}')

#here we find the encodings of the faces which are in our database

encodeListOfKnownFaces = findEncodings(images)
print('Encoding is Done !!')

#cap  = cv2.VideoCapture('video.mp4')
#In cap we capture the current frame of our face cam

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    faceInCurrentFrame = face_recognition.face_locations(imgSmall)
    encodeOfCurrentFrame = face_recognition.face_encodings(imgSmall,faceInCurrentFrame)#128 calculatinos in array

    for encodeFace,faceLoc in zip(encodeOfCurrentFrame,faceInCurrentFrame):
        matches = face_recognition.compare_faces(encodeListOfKnownFaces,encodeFace)
        print(matches)
        faceDis = face_recognition.face_distance(encodeListOfKnownFaces,encodeFace)#euclidean distance
        matchIndex = np.argmin(faceDis)
        print(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)#green
            cv2.rectangle(img,(x1,y2-15),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)#white
            markAttendance(name)
        else:
            print('Unknown Face')
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 15), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, 'Unknown Face', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
# vidSmall = cv2.resize(img,(300,400)) 
# while True: 
#     frames = resize(img, 0.4)
#     cv2.imshow('Video', frames) 
#     if cv2.waitKey(20) & 0xFF == ord('d'): 
#         break

    now = datetime.now()
    timeString = now.strftime('%H::%M::%S')
    cv2.putText(img, "COUNT : " + str(serialNo), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)#blue
    cv2.putText(img, timeString, (460, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)#blue
    cv2.putText(img, "Project", (10, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)#red
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

    #cv2 is used to do things related with images on python
    #basically we are dealing with open cv in python for image related purpose