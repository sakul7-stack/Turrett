#include <Servo.h>

Servo panServo;
Servo tiltServo;

int panPos = 90;
int tiltPos = 90;
char buf[32];

void setup() {
    Serial.begin(9600);
    panServo.attach(9);
    tiltServo.attach(10);
    panServo.write(panPos);
    tiltServo.write(tiltPos);
}

void loop() {
    if (Serial.available()) {
        int len = Serial.readBytesUntil('\n', buf, 31);
        if (len > 0) {
            buf[len] = '\0';
            char* comma = strchr(buf, ',');
            if (comma) {
                *comma = '\0';
                int pd = atoi(buf);
                int td = atoi(comma + 1);

                panPos += pd;
                tiltPos += td;

                panPos = constrain(panPos, 0, 180);
                tiltPos = constrain(tiltPos, 0, 180);

                panServo.write(panPos);
                tiltServo.write(tiltPos);
            }
        }
    }
}