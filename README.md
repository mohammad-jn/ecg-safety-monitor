# ECG Safety Monitor

## 📌 Overview

This project implements a modular ECG analysis pipeline using data from the MIT-BIH Arrhythmia Database. It processes raw ECG signals, detects heartbeats (R-peaks), analyzes rhythm characteristics, and evaluates performance against annotated ground truth.

The system is designed with a focus on **correctness, reliability, and safety**, inspired by medical device software practices.

---

## 🎯 Features

- ECG signal loading from MIT-BIH dataset
- Signal preprocessing
- R-peak detection using signal processing techniques
- Heart rate estimation
- RR interval and heart rate variability (HRV) analysis
- Irregular rhythm detection
- Safety validation layer:
- Ground-truth evaluation:
- Record-level rhythm interpretation
- Full unit test coverage

## 📚 Dataset

MIT-BIH Arrhythmia Database (via PhysioNet / Kaggle)


## ⚙️ Installation

```bash
git clone https://github.com/mohammad-jn/ecg-safety-monitor.git
cd ecg-safety-monitor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## ▶️ Usage

Run the main pipeline:
```bash
python main.py
```

Run tests:
```bash
python -m pytest
```