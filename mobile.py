import cv2, time
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)
adress = "https://192.168.43.1:8082/video"
video.open(adress)

while True:
    check, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=5)
    for x,y,h,w in face:
        img = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 1)
    cv2.imshow('MobileFace', frame)
    key = cv2.waitKey(1)

    if key==ord('q'):
        break

video.release()
cv2.destroyAllWindows()