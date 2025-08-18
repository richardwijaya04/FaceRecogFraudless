Fraudless Attendance: Real-time Face Recognition Service 🧠🎓
Sistem Absensi Cerdas Berbasis AI dengan Pengenalan Wajah dan Registrasi Dinamis

👋 Apa Proyek Ini?
Bayangkan sebuah sistem absensi super canggih yang bisa mengenali wajah mahasiswa secara otomatis melalui kamera. Proyek ini adalah "otak" di balik sistem tersebut. Ini adalah sebuah layanan (service) AI yang berjalan di server, yang bertugas untuk:

Mengenali Wajah: Secara akurat mengidentifikasi siapa yang ada di depan kamera berdasarkan database foto yang sudah ada.

Belajar Wajah Baru: Mampu mempelajari wajah-wajah baru secara langsung (real-time) tanpa perlu mematikan sistem.

Karena berbasis API, "otak" ini siap dihubungkan ke berbagai aplikasi, seperti aplikasi absensi di iOS/Android, aplikasi web, atau sistem keamanan gedung.

🚀 Fitur Utama
Arsitektur Skalabel (Client-Server): Desain backend yang terpisah dari aplikasi pengguna (klien), memungkinkan sistem untuk tumbuh dan melayani banyak pengguna secara efisien.

Identifikasi Real-time: Klien (misalnya webcam) dapat mengirimkan video ke server untuk proses identifikasi wajah yang cepat dan terus-menerus.

Registrasi Dinamis (On-the-Fly): Daftarkan wajah baru kapan saja melalui API. Sistem akan langsung mampu mengenali orang tersebut dalam hitungan detik.

Akurasi Tinggi: Ditenagai oleh model Deep Learning State-of-the-Art dari InsightFace untuk memastikan akurasi pengenalan yang andal.

Siap Integrasi: Fungsionalitas inti diekspos melalui API standar, membuatnya mudah untuk diintegrasikan dengan berbagai platform.

🏛️ Arsitektur Sistem
Sistem ini dirancang dengan pemisahan tugas yang jelas untuk memastikan performa dan kemudahan pengembangan.

AI Backend Service (face_service.py): Sebuah server Flask yang menjadi pusat operasi. Server ini memuat model AI, mengelola database wajah di memori, dan menyediakan API untuk semua permintaan.

Klien (live_client.py & register_client.py): Aplikasi Python yang berkomunikasi dengan server. live_client untuk absensi via webcam, dan register_client untuk menambahkan data wajah baru.

📂 Struktur Folder Proyek
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
🛠️ Instalasi & Konfigurasi
Ikuti langkah-langkah berikut untuk menyiapkan dan menjalankan proyek di komputer Anda.

1. Prasyarat
Python 3.9+

Git

2. Kloning Repositori
Buka terminal Anda dan kloning repositori ini.

Bash

git clone https://github.com/richardwijaya04/FaceRecogFraudless.git
cd FraudlessAttendance2
3. Siapkan Virtual Environment
Menggunakan virtual environment sangat penting agar dependensi proyek tidak tercampur dengan proyek lain.

Bash

# Membuat environment baru bernama 'venv'
python3 -m venv venv

# Mengaktifkan environment (untuk macOS/Linux)
source venv/bin/activate
(Untuk pengguna Windows, gunakan perintah: .\venv\Scripts\activate)

4. Instal Dependensi
Setelah virtual environment aktif, instal semua pustaka yang dibutuhkan.

Bash

pip install -r requirements.txt
Catatan: Pastikan file requirements.txt Anda minimal berisi: flask, requests, opencv-python, insightface, onnxruntime, numpy, pillow.

5. Isi Database Wajah Awal
Sebelum server dijalankan, Anda perlu menyediakan data wajah awal untuk dipelajari. Salin folder-folder siswa (contoh: student_001, student_002, dst.) ke dalam direktori assets/database/.

▶️ Menjalankan Sistem
Sistem ini membutuhkan dua terminal yang berjalan secara bersamaan: satu untuk server dan satu untuk klien.

Langkah 1: Jalankan Server AI (di Terminal 1 🖥️)
Server harus dijalankan terlebih dahulu untuk memuat model dan database ke memori.

Bash

# Pastikan venv sudah aktif di terminal ini
source venv/bin/activate

# Jalankan server dari direktori utama proyek
python scripts/face_service.py
Server sekarang berjalan dan menunggu koneksi di port 5050. Biarkan terminal ini tetap terbuka.

Langkah 2: Jalankan Klien Webcam (di Terminal 2 📹)
Buka jendela terminal BARU.

Bash

# Pindah ke direktori proyek Anda
cd path/to/FraudlessAttendance2

# Aktifkan venv (wajib di setiap terminal baru)
source venv/bin/activate

# Jalankan klien webcam
python scripts/live_client.py
Sebuah jendela akan muncul menampilkan feed webcam Anda, lengkap dengan hasil identifikasi wajah secara real-time.

Langkah 3: Mendaftarkan Wajah Baru (Opsional, di Terminal 2)
Untuk menambahkan wajah baru tanpa mematikan server, gunakan skrip register_client.py di Terminal 2.

Bash

# Format: python scripts/register_client.py <id_baru> "<path/ke/gambar.jpg>"
python scripts/register_client.py student_011 "path/ke/foto/student_011.jpeg"
Setelah perintah ini berhasil, sistem akan langsung mampu mengenali student_011.

⚙️ API Endpoints
Server menyediakan beberapa endpoint untuk interaksi:

Endpoint	Metode	Deskripsi	Body Form-Data
/register	POST	Mendaftarkan wajah baru ke dalam database sistem.	studentId (string), photo (file)
/identify	POST	Mengidentifikasi wajah (1:N) dari gambar yang dikirim.	photo (file)
/verify	POST	Memverifikasi wajah (1:1) dengan ID yang spesifik.	studentId (string), photo (file)
