# Face Recognition Attendance System 🧠🎓

This project is a lightweight face recognition attendance system using [InsightFace](https://github.com/deepinsight/insightface) that processes image files (not webcam) to identify students and record their attendance.

## 🚀 Features
- Input: Single image from file or gallery
- Output: Identity match, similarity score, and face box
- Lightweight and fast (no GPU required)
- Suitable for backend API deployment and mobile integration (Android)
- Evaluation ready: Precision, Recall, F1 Score

## 📂 Folder Structure

├── assets/
│ └── database/
│ └── student_001/
│ └── 1.jpg
├── verify_image.py
├── requirements.txt
└── README.md


## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/face-recognition-attendance.git
cd face-recognition-attendance
pip install -r requirements.txt

## RUN THE SYSTEM
python verify_image.py

## MASUKKAN INPUT SEPERTI INI
Masukkan path gambar (misal: ./assets/test_data/student_008/test_008.jpeg):
