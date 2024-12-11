// switch.h
#ifndef SWITCH_H
#define SWITCH_H

class Switch {
public:
    Switch();                // constructeur
    void start(bool status); 
    void stop();             
    bool getStatus() const;
      

private:
    bool isOn; // status (on/off)
};

#endif
