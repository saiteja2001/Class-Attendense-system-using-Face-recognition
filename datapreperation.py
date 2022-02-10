import imutils # for image processing like rotation,translation,resizing
import time
import cv2  # importing opencv library
import csv  # this module is used to operate csv file.
import os
# this module provides functions for interacting with operating system

cascade = 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade) # loading the classifier

# creating a file with the student name

Name = str(input("Enter your Name : "))
Roll_Number = int(input("Enter your Roll_Number : "))
dataset = 'dataset'
sub_data = Name
path = os.path.join(dataset, sub_data)


if not os.path.isdir(path):
    os.mkdir(path)
    print(sub_data + "file is created.")

# entering the details of student in csv file
info = [str(Name), str(Roll_Number)]
with open('student.csv', 'a') as csvFile:  # open a file to append
    # creating a writer object
    write = csv.writer(csvFile)
    # writing a row
    write.writerow(info)
csvFile.close()

print("Starting video stream...")
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # define object of video captured
time.sleep(2.0)
total = 0

while total < 100 :

    _, frame = cam.read()
    width,height,c=frame.shape
    #img = imutils.resize(frame, width=400)  # resizing the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(
        gray, scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30))    # for detecting the face and eyes
    # convert to grayscale image and reduce the size of 10%(scale factor).
    # it gives the output as dimensions rect(x,y,w,h) of the face

    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # to draw a rectangle on the image frame.color(BGR)
        p = os.path.sep.join([path, "{}.jpg".format(
            str(total).zfill(5))])  # zfill adds zero at the beginning of the story.
        img = gray[y:y + h, x:x + w]
        img = cv2.resize(img, (width, height))
        try:
            cv2.imwrite(p, img)
            total += 1
            print(total)
        except:
            print("face not found")

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF  # it will display the image for one second. the 0xFF does an operation and returns the last 8 bit digits
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
