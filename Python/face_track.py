import cv2
import serial
import numpy as np

def set_res(cap, x, y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))

ser = serial.Serial('COM3', 250000)
cap = cv2.VideoCapture(0)
frame_w = 1280
frame_h = 720
set_res(cap, frame_w, frame_h)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=20,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if len(faces) > 0:
        face_center_x = faces[0][0] + faces[0][2] / 2
        face_center_y = faces[0][1] + faces[0][3] / 2
        err_x = 30 * (face_center_x - frame_w / 2) / (frame_w / 2)
        err_y = 30 * (face_center_y - frame_h / 2) / (frame_h / 2)
        ser.write(f"{err_x}x!".encode())
        ser.write(f"{err_y}y!".encode())
        print("X:", err_x, "Y:", err_y)
    else:
        ser.write("o!".encode())

ser.close()
cap.release()
cv2.destroyAllWindows()