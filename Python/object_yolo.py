import cv2
from ultralytics import YOLO
import serial
import time
import sys


class PID:

    def __init__(self,kp=0.085,ki=0.001,kd=0.035):
        self.kp=kp
        self.ki=ki
        self.kd=kd
        self.prev=0
        self.intg=0

    def update(self,err,dt):
        self.intg+=err*dt
        deriv=(err-self.prev)/dt if dt>0 else 0
        out=self.kp*err+self.ki*self.intg+self.kd*deriv
        self.prev=err
        return out
model=YOLO("yolov8n.pt")


try:
    ser=serial.Serial("COM3",9600,timeout=0.5)

except Exception:
    sys.exit(1)



cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cap.set(cv2.CAP_PROP_FPS,30)
target=input("Enter object to track: ").strip().lower()
pan_pid=PID(0.085,0.001,0.035)
tilt_pid=PID(0.085,0.001,0.035)
last_t=time.time()
pan_pos=90
tilt_pos=90


while True:
    ret,frame=cap.read()
    if not ret:
        break
    results=model(frame,verbose=False,conf=0.35,iou=0.45,max_det=5)
    best=None
    min_d=float("inf")
    h,w=frame.shape[:2]
    cx,cy=w//2,h//2
    for r in results:
        for b in r.boxes:
            cls_name=r.names[int(b.cls[0])].lower()
            if cls_name==target and b.conf[0]>0.35:
                x1,y1,x2,y2=b.xyxy[0].tolist()
                ox=(x1+x2)/2
                oy=(y1+y2)/2
                d=((ox-cx)**2+(oy-cy)**2)**0.5
                if d<min_d:
                    min_d=d
                    best=(ox-cx,oy-cy)

    if best:
        ex,ey=best
        now=time.time()
        dt=now-last_t
        last_t=now
        pd=pan_pid.update(ex,dt)
        td=tilt_pid.update(ey,dt)
        pd=max(min(int(pd),6),-6)
        td=max(min(int(td),6),-6)
        if abs(pd)>0 or abs(td)>0:
            ser.write(f"{pd},{td}\n".encode())
            pan_pos=max(0,min(180,pan_pos+pd))
            tilt_pos=max(0,min(180,tilt_pos+td))
    cv2.imshow("Enterprise Turret Tracker",frame)
    if cv2.waitKey(1)&0xFF==ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
ser.close()