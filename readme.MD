# Fraudless Attendance: Real-time Face Recognition Service 🧠🎓

*Sistem Absensi Cerdas Berbasis AI dengan Pengenalan Wajah dan Registrasi Dinamis*

---

## 👋 Gambaran Umum Proyek

Selamat datang di *Fraudless Attendance System*, sebuah prototipe sistem absensi canggih yang menggunakan pengenalan wajah. Proyek ini berfungsi sebagai "otak" (backend service) yang mampu mengidentifikasi wajah secara *real-time* dan dapat "mempelajari" wajah-wajah baru secara dinamis tanpa perlu dimatikan.

Karena dirancang dengan arsitektur berbasis API, sistem ini sangat fleksibel dan siap untuk dihubungkan dengan berbagai aplikasi, seperti aplikasi absensi pada iOS/Android, website, atau bahkan sistem keamanan gedung.

## 🚀 Fitur Utama

-   **Arsitektur Skalabel (Client-Server):** Desain backend yang terpisah dari aplikasi pengguna (klien), memungkinkan sistem untuk tumbuh dan melayani banyak pengguna secara efisien.
-   **Identifikasi Real-time:** Klien (misalnya webcam) dapat mengirimkan *stream* video ke server untuk proses identifikasi wajah yang cepat dan terus-menerus.
-   **Registrasi Dinamis (On-the-Fly):** Daftarkan wajah baru kapan saja melalui API. Sistem akan langsung mampu mengenali orang tersebut dalam hitungan detik, tanpa perlu restart.
-   **Akurasi Tinggi:** Ditenagai oleh model *Deep Learning State-of-the-Art* dari **InsightFace** untuk memastikan akurasi pengenalan yang andal.
-   **Siap Integrasi:** Fungsionalitas inti diekspos melalui API standar, membuatnya mudah untuk diintegrasikan dengan berbagai platform.

## 💻 Tumpukan Teknologi (Technology Stack)

-   **Backend:** Python, Flask
-   **AI Engine:** InsightFace
-   **Runtime & Libraries:** ONNX Runtime, OpenCV, NumPy, Pillow
-   **Interaksi API:** Requests
-   **Lingkungan:** Virtual Environment (venv)

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

## 🛠️ Panduan Instalasi & Konfigurasi

Ikuti langkah-langkah berikut untuk menyiapkan dan menjalankan proyek di komputer Anda.

### 1. Prasyarat
-   Python 3.9+
-   Git

### 2. Kloning Repositori
Buka terminal Anda, pindah ke direktori kerja Anda, dan kloning repositori ini.
```bash
git clone [https://github.com/richardwijaya04/FaceRecogFraudless.git](https://github.com/richardwijaya04/FaceRecogFraudless.git)
cd FraudlessAttendance2

Tentu. Saya mengerti. Anda membutuhkan satu dokumen final yang merangkum semua yang telah kita bangun—kode yang benar, langkah-langkah yang tepat, dan penjelasan yang jernih—dalam format README yang profesional dan bisa dipahami semua orang.

Saya akan menyaring seluruh riwayat diskusi kita, mengambil setiap potongan kode final, instruksi, dan pelajaran yang kita dapatkan, lalu menyusunnya menjadi dokumen master untuk proyek Anda.

Ini adalah versi definitif yang Anda butuhkan.

Gunakan tombol "Salin" atau "Copy" di pojok blok kode di bawah untuk memastikan formatnya 100% sempurna saat Anda menempelkannya ke file README.md.

Markdown

# Fraudless Attendance: Real-time Face Recognition Service 🧠🎓

*Sistem Absensi Cerdas Berbasis AI dengan Pengenalan Wajah dan Registrasi Dinamis*

---

## 👋 Gambaran Umum Proyek

Selamat datang di *Fraudless Attendance System*, sebuah prototipe sistem absensi canggih yang menggunakan pengenalan wajah. Proyek ini berfungsi sebagai "otak" (backend service) yang mampu mengidentifikasi wajah secara *real-time* dan dapat "mempelajari" wajah-wajah baru secara dinamis tanpa perlu dimatikan.

Karena dirancang dengan arsitektur berbasis API, sistem ini sangat fleksibel dan siap untuk dihubungkan dengan berbagai aplikasi, seperti aplikasi absensi pada iOS/Android, website, atau bahkan sistem keamanan gedung.

## 🚀 Fitur Utama

-   **Arsitektur Skalabel (Client-Server):** Desain backend yang terpisah dari aplikasi pengguna (klien), memungkinkan sistem untuk tumbuh dan melayani banyak pengguna secara efisien.
-   **Identifikasi Real-time:** Klien (misalnya webcam) dapat mengirimkan *stream* video ke server untuk proses identifikasi wajah yang cepat dan terus-menerus.
-   **Registrasi Dinamis (On-the-Fly):** Daftarkan wajah baru kapan saja melalui API. Sistem akan langsung mampu mengenali orang tersebut dalam hitungan detik, tanpa perlu restart.
-   **Akurasi Tinggi:** Ditenagai oleh model *Deep Learning State-of-the-Art* dari **InsightFace** untuk memastikan akurasi pengenalan yang andal.
-   **Siap Integrasi:** Fungsionalitas inti diekspos melalui API standar, membuatnya mudah untuk diintegrasikan dengan berbagai platform.

## 💻 Tumpukan Teknologi (Technology Stack)

-   **Backend:** Python, Flask
-   **AI Engine:** InsightFace
-   **Runtime & Libraries:** ONNX Runtime, OpenCV, NumPy, Pillow
-   **Interaksi API:** Requests
-   **Lingkungan:** Virtual Environment (venv)

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


## 🛠️ Panduan Instalasi & Konfigurasi

Ikuti langkah-langkah berikut untuk menyiapkan dan menjalankan proyek di komputer Anda.

### 1. Prasyarat
-   Python 3.9+
-   Git

### 2. Kloning Repositori
Buka terminal Anda, pindah ke direktori kerja Anda, dan kloning repositori ini.
```bash
git clone [https://github.com/richardwijaya04/FaceRecogFraudless.git](https://github.com/richardwijaya04/FaceRecogFraudless.git)
cd FraudlessAttendance2
3. Siapkan Virtual Environment
Menggunakan virtual environment sangat penting agar dependensi proyek terisolasi dan tidak mengganggu instalasi Python di sistem Anda.

Bash

# Membuat environment baru bernama 'venv'
python3 -m venv venv

# Mengaktifkan environment (untuk macOS/Linux)
source venv/bin/activate
(Untuk pengguna Windows, gunakan perintah: .\venv\Scripts\activate)

4. Instal Dependensi
Setelah virtual environment aktif (ditandai dengan (venv) di awal prompt terminal), instal semua pustaka yang dibutuhkan.

Bash

pip install -r requirements.txt
Catatan: Pastikan file requirements.txt Anda minimal berisi: flask, requests, opencv-python, insightface, onnxruntime, numpy, pillow.

5. Isi Database Wajah Awal
Sebelum server dijalankan, Anda perlu menyediakan data wajah awal untuk dipelajari. Salin folder-folder berisi gambar siswa (contoh: student_001, student_002, dst.) ke dalam direktori assets/database/.

▶️ Cara Menjalankan Sistem
Sistem ini membutuhkan dua terminal yang berjalan secara bersamaan.

Langkah 1: Jalankan Server AI (di Terminal 1 🖥️)
Server adalah "otak" yang harus dijalankan terlebih dahulu.

Bash

# Pastikan venv sudah aktif di terminal ini
# Jika belum, jalankan: source venv/bin/activate

# Jalankan server dari direktori utama proyek
python scripts/face_service.py
Server akan mulai memuat model dan database. Setelah selesai, Anda akan melihat log bahwa server berjalan di port 5050. Biarkan terminal ini tetap terbuka.

Langkah 2: Jalankan Klien Webcam (di Terminal 2 📹)
Buka jendela terminal BARU (ini penting!).

Bash

# Pindah ke direktori proyek Anda
cd path/to/FraudlessAttendance2

# Aktifkan venv (wajib di setiap terminal baru!)
source venv/bin/activate

# Jalankan klien webcam
python scripts/live_client.py
Sebuah jendela akan muncul menampilkan feed dari webcam Anda, lengkap dengan hasil identifikasi wajah secara real-time.

Langkah 3: Mendaftarkan Wajah Baru (di Terminal 2)
Untuk menambahkan wajah baru tanpa mematikan server, gunakan skrip register_client.py di Terminal 2.

Bash

# Format: python scripts/register_client.py <id_baru> "<path/ke/gambar.jpg>"
python scripts/register_client.py student_011 "assets/database/student_011/foto_baru.jpeg"
