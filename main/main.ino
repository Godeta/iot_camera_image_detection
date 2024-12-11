#include "switch.h"

//déclaration
Switch* switch1 = new Switch();
Switch* switch2 = new Switch();

void setup() {
  Serial.begin(9600);
  // put your setup code here, to run once:

  switch1->start(true);
  switch2->start(true);
}

void loop() {
  // put your main code here, to run repeatedly:
  if (switch1->getStatus()){
    switch2->stop();
    Serial.print("ok");
  }
}
