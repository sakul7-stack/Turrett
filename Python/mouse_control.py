import cv2
import serial
import time
import sys
import numpy as np

class PID:
    def __init__(self, kp=0.12, ki=0.002, kd=0.045):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0.001 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

# Configuration
SERIAL_PORT = "COM3"         
BAUDRATE    = 115200
CAM_WIDTH   = 640
CAM_HEIGHT  = 480
DEADZONE_PX = 18
MAX_SPEED   = 7             
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.4)
    time.sleep(1.8)         
except Exception as e:
    print(f"Serial open failed: {e}")
    sys.exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    ser.close()
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

pan_pid  = PID(0.115, 0.0018, 0.038)
tilt_pid = PID(0.115, 0.0018, 0.038)

last_time   = time.time()
current_pan = 90
current_tilt= 90

cv2.namedWindow("Mouse Turret Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mouse Turret Control", CAM_WIDTH, CAM_HEIGHT)

mouse_x = CAM_WIDTH  // 2
mouse_y = CAM_HEIGHT // 2

def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y

cv2.setMouseCallback("Mouse Turret Control", on_mouse)

print("Move mouse in window to control turret. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    dt = now - last_time
    last_time = now

    center_x = CAM_WIDTH  // 2
    center_y = CAM_HEIGHT // 2

    error_x = mouse_x - center_x
    error_y = mouse_y - center_y

    cv2.line(frame, (center_x-20, center_y), (center_x+20, center_y), (0,180,0), 2)
    cv2.line(frame, (center_x, center_y-20), (center_x, center_y+20), (0,180,0), 2)
    cv2.circle(frame, (mouse_x, mouse_y), 8, (0,0,255), -1)
    cv2.circle(frame, (mouse_x, mouse_y), 12, (255,255,0), 2)

    if abs(error_x) > DEADZONE_PX or abs(error_y) > DEADZONE_PX:
        pan_delta  = pan_pid.compute( error_x, dt)
        tilt_delta = tilt_pid.compute(-error_y, dt) 

        pan_step  = max(min(int(round(pan_delta)),  MAX_SPEED), -MAX_SPEED)
        tilt_step = max(min(int(round(tilt_delta)), MAX_SPEED), -MAX_SPEED)

        if pan_step != 0 or tilt_step != 0:
            ser.write(f"{pan_step},{tilt_step}\n".encode())
            current_pan  = max(0, min(180, current_pan  + pan_step))
            current_tilt = max(0, min(180, current_tilt + tilt_step))

    cv2.imshow("Mouse Turret Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
print("Done.")