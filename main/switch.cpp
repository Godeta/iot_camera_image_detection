// switch.cpp
#include "switch.h"

Switch::Switch() : isOn(false) {
    // constructeur
}

void Switch::start(bool status) {
    isOn = status;
    
}

void Switch::stop() {
    isOn = false;
}

bool Switch::getStatus() const {
    return isOn;
}
