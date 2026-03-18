import cv2
import numpy as np
import dlib
from math import hypot


cap=cv2.VideoCapture(0)#webcam
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
font=cv2.FONT_HERSHEY_COMPLEX

def mid_point(p1,p2):
     return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def get_blinking_ratio(eye_points,facial_landmarks):
        left_point=(facial_landmarks.part(eye_points[0]).x,facial_landmarks.part(eye_points[0]).y)
        right_point=(facial_landmarks.part(eye_points[3]).x,facial_landmarks.part(eye_points[3]).y)
        hor_line=cv2.line(frame,left_point,right_point,(0,255,0),2)

        center_top=mid_point(facial_landmarks.part(eye_points[1]),facial_landmarks.part(eye_points[2]))
        center_bottom=mid_point(facial_landmarks.part(eye_points[5]),facial_landmarks.part(eye_points[4]))
        ver_line=cv2.line(frame,center_top,center_bottom,(0,255,0),2)


        ver_line_length=hypot((center_top[0]-center_bottom[0]),(center_top[1]-center_bottom[1]))
        #print(ver_line_length)
        hor_line_length=hypot((left_point[0]-right_point[0]),(left_point[1]-right_point[1]))
        #print(ver_line_length)
        ratio=hor_line_length/ver_line_length
        return ratio

def get_gaze_ratio(eye_points,facial_landmarks):
    left_eye_region=np.array([(facial_landmarks.part(eye_points[0]).x,facial_landmarks.part(eye_points[0]).y),
                                  (facial_landmarks.part(eye_points[1]).x,facial_landmarks.part(eye_points[1]).y),
                                  (facial_landmarks.part(eye_points[2]).x,facial_landmarks.part(eye_points[2]).y),
                                  (facial_landmarks.part(eye_points[3]).x,facial_landmarks.part(eye_points[3]).y),
                                  (facial_landmarks.part(eye_points[4]).x,facial_landmarks.part(eye_points[4]).y),
                                  (facial_landmarks.part(eye_points[5]).x,facial_landmarks.part(eye_points[5]).y)],np.int32)
        #print(left_eye_region)
        #cv2.polylines(frame,[left_eye_region],True,(0,0,255),2)



    height,width,_=frame.shape
    mask=np.zeros((height,width),np.uint8)
    cv2.polylines(mask,[left_eye_region],True,255,2)
    cv2.fillPoly(mask,[left_eye_region],255)
    eye=cv2.bitwise_and(gray,gray,mask=mask)

    min_x=np.min(left_eye_region[:,0])
    max_x=np.max(left_eye_region[:,0])
    min_y=np.min(left_eye_region[:,1])
    max_y=np.max(left_eye_region[:,1])

    gray_eye=eye[min_y:max_y,min_x:max_x]
    #gray_eye=cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
    _,threshold_eye=cv2.threshold(gray_eye,70,255,cv2.THRESH_BINARY_INV)
    height,width=threshold_eye.shape

    left_side_threshold=threshold_eye[0:height,0:int(width/2)]
    left_side_white=cv2.countNonZero(left_side_threshold)

    right_side_threshold=threshold_eye[0:height,int(width/2):width]
    right_side_white=cv2.countNonZero(right_side_threshold)

    """     
    if left_side_white<0.000002:
        gaze_ratio=5
    """
    gaze_ratio=left_side_white/right_side_white
    

    
    return gaze_ratio





        

while True:
    _,frame=cap.read()
    new_frame=np.zeros((500,500,3),np.uint8)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
        print(face)
        #x,y=face.left(),face.top()
        #x1,y1=face.right(),face.bottom()
        #cv2.rectangle(frame,(x,y),(x1,y1),(0,255,0),2)
        landmarks=predictor(gray,face)
        #print(landmarks)
        #print(landmarks.part(36))
        #x=landmarks.part(36).x
        #y=landmarks.part(36).y
        #cv2.circle(frame,(x,y),3,(0,0,255),1)
        left_eye_ratio=get_blinking_ratio([36,37,38,39,40,41],landmarks)
        right_eye_ratio=get_blinking_ratio([42,43,44,45,46,47],landmarks)
        blinking_ratio=(left_eye_ratio+right_eye_ratio)/2
        
    

        if blinking_ratio>5.7:
            cv2.putText(frame,'blinking',(50,150),font,3,(255,0,0))



        #gaze detection
        gaze_ratio_left_eye=get_gaze_ratio([36,37,38,39,40,41],landmarks)
        gaze_ratio_right_eye=get_gaze_ratio([42,43,44,45,46,47],landmarks)
        gaze_ratio=(gaze_ratio_left_eye+gaze_ratio_right_eye)/2

        

        if gaze_ratio<=0.58:
             cv2.putText(frame,"LEFT",(50,100),font,2,(0,0,255),3)
             new_frame[:]=(0,0,255)
        elif 0.4<gaze_ratio<1:
             cv2.putText(frame,"center",(50,100),font,2,(0,0,255),3)
        else:
             new_frame[:]=(255,0,0)
             cv2.putText(frame,"right",(50,100),font,2,(0,0,255),3)



            ##showing derection
        #new_frame=np.zeros((500,500,3),np.uint8)
             
             
        


        #threshold_eye=cv2.resize(threshold_eye,None,fx=5,fy=5)
        #eye=cv2.resize(gray_eye,None,fx=5,fy=5)
        #cv2.putText(frame,str(left_side_white),(50,100),font,2,(0,0,255),3)
        #cv2.putText(frame,str(right_side_white),(50,150),font,2,(0,0,255),3)
        cv2.putText(frame,str(gaze_ratio),(50,150),font,2,(0,0,255),3)

        
        #cv2.imshow("eye",eye)
        #cv2.imshow("left_eye",left_eye)
        #cv2.imshow('threshold',threshold_eye)
        #cv2.imshow("left",left_side_threshold)
        #cv2.imshow("right",right_side_threshold)


       


    cv2.imshow("frame",frame)
    cv2.imshow('new_frame',new_frame)
    key=cv2.waitKey(1)
    if key==27:#esc
        break

cap.release()
cv2.destroyAllWindows()