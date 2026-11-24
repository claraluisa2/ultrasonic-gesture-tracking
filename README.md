
# Ultrasonic Gesture Tracking

Gesture tracking and motion analysis using two ultrasonic sensors and an ESP32-S3.

## About the Project

This project explores motion tracking and gesture recognition
using two ultrasonic sensors positioned perpendicularly (L-shaped
configuration).

The system collects distance measurements from both sensors
and stores the data in CSV files for further analysis.

## Hardware

- ESP32-S3 DevKitC-1
- 2 ultrasonic distance sensors

## Technologies

- C++
- Python
- PlatformIO
- Pandas
- Matplotlib

## Project Structure

```text
.
├── dados/
├── include/
├── lib/
├── scripts/
│   ├── coletar_dados.py
│   └── graficos.py
├── src/
│   └── main.cpp
├── test/
├── platformio.ini
└── README.md
```

## Data Collection

The system records:

- Timestamp
- Relative time
- Sensor 1 distance (d1)
- Sensor 2 distance (d2)
- Gesture label
- Person identifier
- Object identifier

## Data Visualization

Python scripts are used to visualize the
measurements and analyze the movement trajectory.

## Current Status

- [x] Ultrasonic sensor integration
- [x] Data acquisition
- [x] CSV data logging
- [x] Initial data visualization
- [ ] Trajectory analysis
- [ ] Gesture recognition

## Future Improvements

- Noise filtering
- Trajectory reconstruction
- Gesture classification

## Author

Computer Engineering student
