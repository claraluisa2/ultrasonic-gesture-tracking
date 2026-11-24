#include <Arduino.h>

// HC-SR04 nº 1
#define TRIG1 5
#define ECHO1 18

// HC-SR04 nº 2
#define TRIG2 6
#define ECHO2 7

#define TIMEOUT 30000

// Mede a distância em cm; retorna -1 se não houver leitura válida
float lerDistancia(int trigPin, int echoPin) {

    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);

    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);

    digitalWrite(trigPin, LOW);

    unsigned long duracao = pulseIn(echoPin, HIGH, TIMEOUT);

    if (duracao == 0) {
        return -1;
    }

    float distancia = duracao / 58.0;

    if (distancia < 2.0 || distancia > 400.0) {
        return -1;
    }

    return distancia;
}

void setup() {

    Serial.begin(115200);

    pinMode(TRIG1, OUTPUT);
    pinMode(ECHO1, INPUT);

    pinMode(TRIG2, OUTPUT);
    pinMode(ECHO2, INPUT);

    digitalWrite(TRIG1, LOW);
    digitalWrite(TRIG2, LOW);
}

void loop() {

    float distancia1 = lerDistancia(TRIG1, ECHO1);

    // Intervalo para o eco do sensor 1 sumir antes de disparar o 2
    delay(30);

    float distancia2 = lerDistancia(TRIG2, ECHO2);

    // Formato esperado pelo script Python: tempo_ms,d1,d2
    Serial.printf("%lu,%.1f,%.1f\n", millis(), distancia1, distancia2);

    delay(30);
}