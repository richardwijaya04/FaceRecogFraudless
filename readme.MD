# Fraudless Attendance: Real-time Face Recognition Service 🧠🎓

Selamat datang di *Fraudless Attendance System*, sebuah sistem pengenalan wajah berbasis AI yang dirancang sebagai layanan (*service*) yang kokoh dan dapat diskalakan. Proyek ini menggunakan model *deep learning* canggih dari **InsightFace** untuk menyediakan identifikasi wajah secara *real-time* dan kemampuan untuk mendaftarkan identitas baru secara dinamis tanpa perlu mematikan sistem.

## 🚀 Fitur Utama

-   **Arsitektur Client-Server:** Sistem dipisahkan menjadi Backend AI (server) dan Klien, memungkinkan skalabilitas dan fleksibilitas.
-   **Identifikasi Real-time:** Klien webcam dapat mengirimkan *stream* video ke server untuk proses identifikasi wajah secara terus-menerus.
-   **Registrasi Dinamis (On-the-Fly):** Daftarkan wajah baru ke dalam database kapan saja melalui API endpoint, dan sistem akan langsung mampu mengenalinya tanpa perlu di-restart.
-   **Akurasi Tinggi:** Ditenagai oleh model SOTA (*State-of-the-Art*) dari InsightFace untuk memastikan akurasi pengenalan yang tinggi.
-   **Berbasis API:** Semua fungsionalitas inti diekspos melalui API, siap untuk diintegrasikan dengan aplikasi web, mobile, atau layanan lainnya.

## 🏛️ Arsitektur Sistem

Sistem ini dirancang dengan pemisahan tugas yang jelas untuk memastikan performa dan kemudahan pengembangan.

1.  **AI Backend Service (`face_service.py`):**
    * Dibangun menggunakan **Flask**.
    * Bertugas memuat model AI ke memori, mengelola database wajah, dan menyediakan API untuk operasi pengenalan dan registrasi.
    * Ini adalah "otak" dari keseluruhan sistem.

2.  **Klien (`live_client.py` & `register_client.py`):**
    * Aplikasi Python yang berinteraksi dengan server.
    * `live_client.py` menangkap video dari webcam dan meminta identifikasi.
    * `register_client.py` digunakan untuk menambahkan wajah baru ke sistem.

## 📂 Struktur Folder Proyek

FraudlessAttendance2/
|
+-- assets/
|   |
|   +-- database/       # <-- Tempat menyimpan gambar referensi (training)
|       |
|       +-- student_001/
|           |
|           +-- 001_1.jpg
|           +-- ...
|
+-- scripts/            # <-- Semua kode Python ada di sini
|   |
|   +-- face_service.py     # (Backend Server AI)
|   +-- live_client.py      # (Klien Webcam Real-time)
|   +-- register_client.py  # (Klien untuk mendaftarkan wajah baru)
|
+-- venv/                 # (Folder Virtual Environment)
|
+-- requirements.txt      # (Daftar dependensi)


## 🛠️ Instalasi & Konfigurasi

Ikuti langkah-langkah berikut untuk menyiapkan dan menjalankan proyek.

#### 1. Prasyarat
-   Python 3.9+
-   Git

#### 2. Kloning Repositori
Buka terminal Anda dan kloning repositori ini.
```bash
git clone [https://github.com/richardwijaya04/FaceRecogFraudless.git](https://github.com/richardwijaya04/FaceRecogFraudless.git)
cd FraudlessAttendance2
3. Siapkan Virtual Environment
Sangat penting untuk menggunakan virtual environment agar dependensi proyek terisolasi.

Bash

# Membuat environment baru bernama 'venv'
python3 -m venv venv

# Mengaktifkan environment (macOS/Linux)
source venv/bin/activate
(Untuk Windows, gunakan: .\venv\Scripts\activate)

4. Instal Dependensi
Pastikan file requirements.txt Anda berisi semua pustaka yang dibutuhkan, lalu jalankan:

Bash

pip install -r requirements.txt
Catatan: Pastikan file requirements.txt Anda minimal berisi: flask, requests, opencv-python, insightface, onnxruntime, numpy, pillow.

5. Isi Database Wajah Awal
Sebelum menjalankan server, isi folder assets/database/ dengan data wajah awal. Buat satu subfolder untuk setiap orang, dan letakkan beberapa contoh fotonya di dalamnya (sesuai struktur di atas).

▶️ Menjalankan Sistem
Sistem ini membutuhkan dua terminal yang berjalan secara bersamaan: satu untuk server dan satu untuk klien.

Langkah 1: Jalankan Server AI (di Terminal 1 🖥️)
Server harus dijalankan terlebih dahulu. Ia akan memuat model dan database ke memori.

Bash

# Pastikan venv sudah aktif
source venv/bin/activate

# Jalankan server dari direktori utama
python scripts/face_service.py
Server sekarang berjalan dan menunggu koneksi di port 5050. Biarkan terminal ini tetap terbuka.

Langkah 2: Jalankan Klien Webcam (di Terminal 2 📹)
Buka jendela terminal BARU.

Bash

# Pindah ke direktori proyek
cd path/to/FraudlessAttendance2

# Aktifkan venv (wajib di setiap terminal baru)
source venv/bin/activate

# Jalankan klien webcam
python scripts/live_client.py
Sebuah jendela akan muncul menampilkan feed webcam Anda, lengkap dengan hasil identifikasi wajah secara real-time.

Langkah 3: Mendaftarkan Wajah Baru (di Terminal 2)
Untuk menambahkan wajah baru tanpa mematikan server, gunakan skrip register_client.py di Terminal 2.

Bash

# Format: python scripts/register_client.py <id_baru> "<path/ke/gambar.jpg>"
python scripts/register_client.py student_011 "path/ke/foto/student_011.jpeg"
Setelah perintah ini berhasil, live_client.py akan langsung mampu mengenali student_011.

⚙️ API Endpoints
Server menyediakan beberapa endpoint untuk interaksi:

Endpoint	Metode	Deskripsi	Body Form-Data
/register	POST	Mendaftarkan wajah baru ke dalam database sistem.	studentId (string), photo (file)
/identify	POST	Mengidentifikasi wajah (1:N) dari gambar yang dikirim.	photo (file)
/verify	POST	Memverifikasi wajah (1:1) dengan ID yang spesifik.	studentId (string), photo (file)
