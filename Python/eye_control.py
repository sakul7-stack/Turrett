import cv2
import numpy as np
import dlib
from math import hypot
import serial
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
font = cv2.FONT_HERSHEY_COMPLEX

def mid_point(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_blinking_ratio(eye_points, landmarks):
    left = (landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y)
    right = (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y)
    center_top = mid_point(landmarks.part(eye_points[1]), landmarks.part(eye_points[2]))
    center_bottom = mid_point(landmarks.part(eye_points[5]), landmarks.part(eye_points[4]))
    hor_len = hypot(left[0] - right[0], left[1] - right[1])
    ver_len = hypot(center_top[0] - center_bottom[0], center_top[1] - center_bottom[1])
    return hor_len / ver_len if ver_len != 0 else 1.0

def get_gaze_ratio(eye_points, landmarks, frame, gray):
    points = [landmarks.part(i) for i in eye_points]
    eye_region = np.array([(p.x, p.y) for p in points], np.int32)
    
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    
    min_x, max_x = np.min(eye_region[:,0]), np.max(eye_region[:,0])
    min_y, max_y = np.min(eye_region[:,1]), np.max(eye_region[:,1])
    gray_eye = eye[min_y:max_y, min_x:max_x]
    
    if gray_eye.size == 0:
        return 1.0
    
    _, thresh = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)
    h, w = thresh.shape
    

    left_side = thresh[0:h, 0:int(w/2)]
    right_side = thresh[0:h, int(w/2):w]
    left_white = cv2.countNonZero(left_side)
    right_white = cv2.countNonZero(right_side)
    hor_ratio = left_white / (right_white + 1e-6)

    top_side = thresh[0:int(h/2), 0:w]
    bottom_side = thresh[int(h/2):h, 0:w]
    top_white = cv2.countNonZero(top_side)
    bottom_white = cv2.countNonZero(bottom_side)
    ver_ratio = top_white / (bottom_white + 1e-6)
    
    return hor_ratio, ver_ratio

try:
    ser = serial.Serial("COM3", 115200, timeout=0.1)
    time.sleep(2.0)
    print("Serial connected")
except Exception as e:
    ser = None
    print(f"Serial failed: {e} → running in display-only mode")

last_send_time = time.time()
send_interval = 0.08  

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    pan_step = 0
    tilt_step = 0
    status = "CENTER"
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Blink detection 
        left_blink = get_blinking_ratio([36,37,38,39,40,41], landmarks)
        right_blink = get_blinking_ratio([42,43,44,45,46,47], landmarks)
        blink_ratio = (left_blink + right_blink) / 2
        if blink_ratio > 5.7:
            cv2.putText(frame, "BLINK", (50, 180), font, 2, (0, 0, 255), 3)
        
        # Gaze ratios 
        h_left, v_left = get_gaze_ratio([36,37,38,39,40,41], landmarks, frame, gray)
        h_right, v_right = get_gaze_ratio([42,43,44,45,46,47], landmarks, frame, gray)
        
        hor_gaze = (h_left + h_right) / 2
        ver_gaze = (v_left + v_right) / 2
        
        # Horizontal 
        if hor_gaze < 0.65:
            pan_step = -4   
            status = "LEFT"
        elif hor_gaze > 1.5:
            pan_step = 4      
            status = "RIGHT"
        else:
            pan_step = 0
        
        # Vertical (tilt)
        if ver_gaze > 1.6:   
            tilt_step = -4    
            status += " + UP"
        elif ver_gaze < 0.7:  
            tilt_step = 4
            status += " + DOWN"
        else:
            tilt_step = 0

        cv2.putText(frame, f"H:{hor_gaze:.2f} V:{ver_gaze:.2f}", (50, 50), font, 1, (0,255,0), 2)
        cv2.putText(frame, status, (50, 100), font, 2, (0, 165, 255), 3)
    

    now = time.time()
    if ser and (pan_step != 0 or tilt_step != 0) and (now - last_send_time > send_interval):
        ser.write(f"{pan_step},{tilt_step}\n".encode())
        last_send_time = now
    
    cv2.imshow("Eye Gaze Turret Control", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()