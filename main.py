import cv2, numpy, os, csv
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'dataset'
print('Training...')
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
            #print(labels)
        id += 1
(width, height) = (130, 100)

(images, labels) = [numpy.array(lis) for lis in [images, labels]]

#print(images, labels)
model = cv2.face.LBPHFaceRecognizer_create()
#model =  cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cnt=0
attendence=set()

while True:
    (_, im) = webcam.read()
    width, height, c = im.shape
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if prediction[1]<60:
            cv2.putText(im,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            print (names[prediction[0]],prediction[1])
            attendence.add(prediction[0])
            cnt=0
        else:
            cnt+=1
            cv2.putText(im,'Unknown',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            if(cnt>100):
                print("Unknown Person")
                #cv2.imwrite("input.jpg",im)
                cnt=0
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10) 
    if key == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()


with open("attendencelist.csv", "a") as csvFile:
    write = csv.writer(csvFile)
    for i in attendence:
        x=[i,names[i]]
        write.writerow(x)
csvFile.close()


