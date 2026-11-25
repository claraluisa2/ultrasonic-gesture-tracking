#include <Arduino.h>

// HC-SR04 nº 1  (mede o eixo X)
#define TRIG1 5
#define ECHO1 18   // ATENÇÃO: o ECHO do HC-SR04 sai em 5 V -> use divisor 1k/2k antes do ESP32-S3

// HC-SR04 nº 2  (mede o eixo Y)
#define TRIG2 6
#define ECHO2 7    // idem: divisor de tensão

// Alcance útil ~40 cm. Eco que demora mais que isso vira "leitura inválida" (-1),
// o que descarta paredes e objetos de fundo em vez de tratá-los como gesto.
#define DIST_MIN_CM 2.0f
#define DIST_MAX_CM 100.0f
#define TIMEOUT_US 6000UL // ida e volta de ~43 cm

#define PERIODO_MS 70           // uma amostra completa (d1 e d2) a cada 70 ms (~14 Hz)
#define INTERVALO_SENSORES_MS 30  // deixa o eco de um sensor morrer antes de disparar o outro

// Mede a distância em cm; retorna -1 se não houver leitura válida
float lerDistancia(int trigPin, int echoPin) {
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);

    unsigned long duracao = pulseIn(echoPin, HIGH, TIMEOUT_US);
    if (duracao == 0) {
        return -1;
    }

    float distancia = duracao / 58.0f;
    if (distancia < DIST_MIN_CM || distancia > DIST_MAX_CM) {
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
    // Período fixo: o intervalo entre amostras fica constante (~70 ms),
    // o que importa porque o tempo é uma feature da rede neural.
    static uint32_t proxima = millis();

    uint32_t t = millis();
    float d1 = lerDistancia(TRIG1, ECHO1);
    delay(INTERVALO_SENSORES_MS);
    float d2 = lerDistancia(TRIG2, ECHO2);

    // Formato esperado pelos scripts Python: tempo_ms,d1,d2  (-1 = leitura inválida)
    Serial.printf("%lu,%.1f,%.1f\n", (unsigned long)t, d1, d2);

    proxima += PERIODO_MS;
    int32_t resta = (int32_t)(proxima - millis());
    if (resta > 0) {
        delay(resta);
    } else {
        proxima = millis();  // atrasou: ressincroniza em vez de acumular atraso
    }
}