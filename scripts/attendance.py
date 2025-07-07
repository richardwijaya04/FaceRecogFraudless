import time
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
import os
from datetime import datetime
import logging

# Setup logging untuk debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inisialisasi InsightFace
try:
    app = FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=0, det_size=(640, 640))  # Resolusi besar untuk akurasi
    logger.info("InsightFace berhasil diinisialisasi")
except Exception as e:
    logger.error(f"Gagal inisialisasi InsightFace: {e}")
    exit(1)

# Path untuk database dan output
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "../assets/database")
OUTPUT_PATH = os.path.join(BASE_DIR, "../output/attendance_log.csv")

# Fungsi untuk memuat database wajah
def load_face_database():
    database = {}
    if not os.path.exists(DATABASE_PATH):
        logger.error(f"Folder database tidak ditemukan: {DATABASE_PATH}")
        return database
    
    for student_id in os.listdir(DATABASE_PATH):
        student_path = os.path.join(DATABASE_PATH, student_id)
        if os.path.isdir(student_path):
            embeddings = []
            for img_name in os.listdir(student_path):
                img_path = os.path.join(student_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"Gagal membaca gambar: {img_path}")
                    continue
                faces = app.get(img)
                if len(faces) == 0:
                    logger.warning(f"Tidak ada wajah terdeteksi di: {img_path}")
                    continue
                embeddings.append(faces[0].embedding)
            if embeddings:
                database[student_id] = np.mean(embeddings, axis=0)
                logger.info(f"Loaded embeddings untuk mahasiswa: {student_id}")
            else:
                logger.warning(f"Tidak ada embeddings valid untuk: {student_id}")
    return database

# Fungsi untuk verifikasi wajah
def verify_face(embedding, database, threshold=0.6):
    if not database:
        return None, float('inf')
    
    min_dist = float('inf')
    identity = None
    try:
        for student_id, db_embedding in database.items():
            dist = 1 - (embedding @ db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
            if dist < min_dist:
                min_dist = dist
                identity = student_id
        if min_dist < threshold:
            return identity, min_dist
        return None, min_dist
    except Exception as e:
        logger.error(f"Error verifikasi wajah: {e}")
        return None, float('inf')

# Fungsi untuk simpan log absensi
def save_attendance(student_id, status="Present"):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log = pd.DataFrame([[student_id, timestamp, status]], 
                          columns=["Student_ID", "Timestamp", "Status"])
        if os.path.exists(OUTPUT_PATH):
            log.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
        else:
            log.to_csv(OUTPUT_PATH, mode='w', header=True, index=False)
        logger.info(f"Absensi tersimpan: {student_id} - {status}")
    except Exception as e:
        logger.error(f"Error menyimpan absensi: {e}")

# Fungsi utama
def main():
    # Muat database
    logger.info("Memuat database wajah...")
    database = load_face_database()
    if not database:
        logger.error("Database kosong! Pastikan folder assets/database berisi gambar.")
        return
    
    # Buka webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Gagal membuka webcam!")
        return
    
    # Variabel untuk verifikasi
    last_verified_id = None
    no_face_frames = 0
    verification_timeout = time.time()
    VERIFICATION_INTERVAL = 5  # Detik sebelum verifikasi ulang

    logger.info("Sistem absensi dimulai. Arahkan wajah ke kamera. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Gagal membaca frame dari webcam!")
            break
        
        # Deteksi wajah
        try:
            faces = app.get(frame)
        except Exception as e:
            logger.error(f"Error deteksi wajah: {e}")
            faces = []
        
        if len(faces) == 0:
            no_face_frames += 1
            if no_face_frames >= 30:  # Reset setelah ~1 detik tanpa wajah (30 frames @ 30 FPS)
                last_verified_id = None
                verification_timeout = time.time()
                logger.info("Reset verifikasi: Tidak ada wajah terdeteksi")
            cv2.putText(frame, "Tidak ada wajah terdeteksi", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            no_face_frames = 0  # Reset counter saat wajah terdeteksi
            for face in faces:
                # Gambar kotak wajah
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Verifikasi wajah
                current_time = time.time()
                if current_time - verification_timeout >= VERIFICATION_INTERVAL or last_verified_id is None:
                    identity, distance = verify_face(face.embedding, database)
                    if identity:
                        cv2.putText(frame, f"Verified: {identity} (Dist: {distance:.2f})", 
                                   (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        if identity != last_verified_id:
                            save_attendance(identity)
                            last_verified_id = identity
                            verification_timeout = current_time
                            logger.info(f"Mahasiswa terverifikasi: {identity}, Distance: {distance:.2f}")
                    else:
                        cv2.putText(frame, f"Tidak dikenali (Dist: {distance:.2f})", 
                                   (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        logger.info(f"Wajah tidak dikenali, Distance: {distance:.2f}")
                else:
                    # Tampilkan ID terakhir kalau masih dalam interval
                    if last_verified_id:
                        cv2.putText(frame, f"Verified: {last_verified_id} (Last Dist)", 
                                   (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Verifikasi tertunda", 
                                   (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Tampilkan frame
        cv2.imshow("Fraudless Attendance System", frame)
        
        # Keluar dengan 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Bersihkan
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Sistem absensi selesai.")

if __name__ == "__main__":
    main()