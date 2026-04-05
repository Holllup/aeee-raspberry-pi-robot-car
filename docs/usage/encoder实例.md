#include <Wire.h>


#define buttonPin 11
#define encoder_shift 0x80000000 //encoder counter is integer number, initial value is 0x80000000 to let value goes up and down (motor goes forward and backward)
#define TEN_METER (1112)   // 234.256*1/(0.067*3.1415926)
long unsigned int encoder1Value = 0;
long unsigned int encoder2Value = 0;
long unsigned int encoder3Value = 0;
long unsigned int encoder4Value = 0;

long unsigned int target1 = 0;

void setup() {
  unsigned char i=0;
  // put your setup code here, to run once:
  pinMode(13, OUTPUT);  // Configure the reset pin as an output and hold the robot in reset
  pinMode(buttonPin, INPUT_PULLUP);  // Enable internal pull-up resistor
  digitalWrite(13,LOW);
  
  //Setup the serial ports
  Serial.begin(57600);    // This is the hardware port connected to the PC through the USB connection
  Wire.begin();
  target1 = encoder_shift + TEN_METER;

  // wait 5 seconds
  Serial.println("Press the button to start"); 
  while (digitalRead(buttonPin) == HIGH) {
    // Do nothing until button is pressed
    delay(100);  // Small delay to avoid excessive CPU usage
  }
  Serial.println("Program started!");  // Send some text to the PC
  delay(5000);
  Serial.println("software serial simple test!");  // Send some text to the PC
  
  digitalWrite(13,HIGH);  // Release the robot from reset
  delay(100);  // A short delay to allow the robot to start-up
 


  Wire.beginTransmission(42);
  Wire.write("sa");
  for(i=0;i<=3;i++)
  {
    Wire.write(20);
    Wire.write(0);
  }
  Wire.endTransmission();
}

void loop() {
  // put your main code here, to run repeatedly:
  
  delay(10);
  readEncoder();
  Serial.println("Encoder 1 read text!");  
  Serial.println(encoder1Value-encoder_shift);  // Number of pulses in encoder 1 through one channel
  Serial.println("Encoder 2 read text!");  
  Serial.println(encoder2Value-encoder_shift);  // Number of pulses in encoder 2 through one channel
  Serial.println("Encoder 3 read text!");  
  Serial.println(encoder3Value-encoder_shift);  // Number of pulses in encoder 3 through one channel
  Serial.println("Encoder 4 read text!");  
  Serial.println(encoder4Value-encoder_shift);  // Number of pulses in encoder 4 through one channel
  delay(500);
  
  if (encoder1Value > 0x80040000)
  {
    encoder1Value=0x80000000;
  }
  else if (encoder1Value < 0x7FFC0000)
  {
    encoder1Value=0x80000000;
  }
  
  if (encoder1Value> target1)
  {
    //Stop all the motors
    Wire.beginTransmission(42);
    Wire.write("ha");
    Wire.endTransmission();
    digitalWrite(13,HIGH);
    while(1);
  }
}

void readEncoder()
{
  long unsigned int encoder1 = 0;
  long unsigned int encoder2 = 0;
  long unsigned int encoder3 = 0;
  long unsigned int encoder4 = 0;
  Wire.beginTransmission(42);
  Wire.write("i0");
  Wire.endTransmission();
  delay(1);
  Wire.requestFrom(42,8);
  delay(10); 
  //if(Wire.available()==8)
  {
    encoder1 = (long unsigned int) Wire.read();

    encoder1 += ((long unsigned int) Wire.read() <<8);
    encoder1 += ((long unsigned int) Wire.read() <<16);
    encoder1 += ((long unsigned int) Wire.read() <<24);
    encoder2 = (long unsigned int) Wire.read();
    encoder2 += ((long unsigned int) Wire.read() <<8);
    encoder2 += ((long unsigned int) Wire.read() <<16);
    encoder2 += ((long unsigned int) Wire.read() <<24);
  }
  encoder1Value = encoder1;
  encoder2Value = encoder2;
    Wire.beginTransmission(42);
  Wire.write("i5");
  Wire.endTransmission();
  delay(1);
  Wire.requestFrom(42,8);
  delay(10);
  //if(Wire.available()==8)
  {
    encoder3 = (long unsigned int) Wire.read();
    encoder3 += ((long unsigned int) Wire.read() <<8);
    encoder3 += ((long unsigned int) Wire.read() <<16);
    encoder3 += ((long unsigned int) Wire.read() <<24);
    encoder4 = (long unsigned int) Wire.read();
    encoder4 += ((long unsigned int) Wire.read() <<8);
    encoder4 += ((long unsigned int) Wire.read() <<16);
    encoder4 += ((long unsigned int) Wire.read() <<24);
  }
  encoder3Value = encoder3;
  encoder4Value = encoder4;  
}


