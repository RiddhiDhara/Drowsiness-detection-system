# Drowsiness Detection System

![Drowsiness Detection Demo](assets/demo.gif)

A real-time fatigue monitoring system combining Python for computer vision and Electron for cross-platform GUI.

## Features

* 👁 **Eye tracking** (blink detection using EAR)
* 👄 **Mouth detection** (yawn detection using MAR)
* 🧠 **Head pose estimation** (nodding/tilting detection)
* 🚨 **3-level alert system** (visual/audio/emergency)
* 📊 **Data logging** (CSV export with timestamps)
* 🖥 **Electron GUI** with live camera preview

## Installation

### Prerequisites

* Python 3.8+
* Node.js 16+
* Webcam

### Setup

```bash
# Clone repository
git clone https://github.com/RiddhiDhara/Drowsiness-detection-system.git
cd Drowsiness-detection-system

# Install Python dependencies
pip install -r requirements.txt

# Install Node modules
npm install
```

### Usage

#### Development

```bash
# In separate terminal, start Electron
npx electron .
```

## Project Structure

```
.
├── node_modules/         # Dependencies
├── python/               # Python scripts
│   ├── 1.py              # Drowsiness detection script
│   ├── 2.py              # Data processing script
│   ├── drowsiness_data.csv # Collected data
│   └── stat.txt          # Statistical data
├── index.html            # Main HTML file
├── main.js               # Electron main process
├── renderer.js           # Frontend rendering script
├── script.js             # Additional JS logic
├── style.css             # Styling file
├── package.json          # Project configuration
├── package-lock.json     # Dependency lock file
└── README.md             # Project documentation
```

## Thresholds 

``` 
"thresholds": {
    "ear_threshold": 0.25,
    "mar_threshold": 0.6,
    "closed_eye_frames": 30,
    "long_blink_threshold": 0.4,
    "head_movement_threshold": 0.2,
    "warning_duration": {
        "single_sign": 5,
        "multiple_signs": 10,
        "drowsiness": 15
    },
    "drowsiness_time_window": 600,
    "sign_combination_threshold": 3
  },
  "alerts": {
    "visual": true,
    "audio": true,
    "emergency_delay_sec": 10
  }
```

## Contact

🔗 [GitHub](https://github.com/RiddhiDhara)
