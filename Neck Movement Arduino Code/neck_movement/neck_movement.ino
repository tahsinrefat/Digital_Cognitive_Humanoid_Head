#include<Servo.h>
Servo servoX;
Servo servoY;
int x = 0;
int y = 0;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  servoY.attach(9);
  servoX.attach(11);

  servoX.write(x);
  servoY.write(y);
  delay(1000);
  
}

char input  = "";

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available()){
    input = Serial.read();
    if (input=='D'){
      servoY.write(y+3);
      y+=3;
    }
    else if(input=='U'){
      servoY.write(y-3);
      y-=3;
    }
    else{
      servoY.write(y);
    }
    if(input=='L'){
      servoX.write(x+3);
      x+=3;
    }
    else if(input=='R'){
      servoX.write(x-3);
      x-=3;
    }
    else{
      servoX.write(x);
    }
    input = "";
  }

}
