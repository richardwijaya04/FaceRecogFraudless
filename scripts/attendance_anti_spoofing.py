import time
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
import os
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inisialisasi InsightFace dengan landmark
try:
    app = FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition', 'landmark_2d_106'])
    app.prepare(ctx_id=0, det_size=(320, 320))  # Resolusi kecil untuk Mac
    logger.info("InsightFace berhasil diinisialisasi")
except Exception as e:
    logger.error(f"Gagal inisialisasi InsightFace: {e}")
    exit(1)

# Path untuk database dan output
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "../assets/database")
OUTPUT_PATH = os.path.join(BASE_DIR, "../output/attendance_log.csv")
PLOT_PATH = os.path.join(BASE_DIR, "../output/attendance_plot.png")

# Fungsi untuk preprocess gambar
def preprocess_image(img):
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max(320 / h, 320 / w)
    if scale > 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    return img

# Fungsi untuk memuat database wajah
def load_face_database():
    database = {}
    if not os.path.exists(DATABASE_PATH):
        logger.error(f"Folder database tidak ditemukan: {DATABASE_PATH}")
        return database
    
    for student_id in os.listdir(DATABASE_PATH):
        if student_id.startswith('.'):
            continue
        student_path = os.path.join(DATABASE_PATH, student_id)
        if os.path.isdir(student_path):
            embeddings = []
            for img_name in os.listdir(student_path):
                if img_name.startswith('.'):
                    continue
                img_path = os.path.join(student_path, img_name)
                img = cv2.imread(img_path)
                img = preprocess_image(img)
                if img is None:
                    logger.warning(f"Gagal membaca gambar: {img_path}")
                    continue
                faces = app.get(img)
                if len(faces) == 0:
                    logger.warning(f"Tidak ada wajah terdeteksi di: {img_path}")
                    continue
                face = faces[0]
                if not hasattr(face, 'embedding') or face.embedding is None:
                    logger.warning(f"Embedding tidak tersedia di: {img_path}")
                    continue
                embeddings.append(face.embedding)
                logger.info(f"Berhasil memproses: {img_path}, ukuran: {img.shape}")
            if embeddings:
                database[student_id] = np.mean(embeddings, axis=0)
                logger.info(f"Loaded embeddings untuk mahasiswa: {student_id}")
            else:
                logger.warning(f"Tidak ada embeddings valid untuk: {student_id}")
    return database

# Fungsi untuk menghitung rasio aspek mata (EAR)
def eye_aspect_ratio(eye_landmarks):
    try:
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        ear = (A + B) / (2.0 * C)
        return ear
    except Exception as e:
        logger.error(f"Error menghitung EAR: {e}")
        return 0.0

# Fungsi untuk deteksi kedipan
def detect_blink(face, blink_history, ear_threshold=0.15, min_blinks=2, max_frames=60):
    if not hasattr(face, 'landmark_2d_106') or face.landmark_2d_106 is None:
        logger.warning("Landmark tidak tersedia untuk deteksi kedipan")
        return False, 0.0
    
    left_eye = face.landmark_2d_106[33:39]
    right_eye = face.landmark_2d_106[94:100]
    
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0 if left_ear > 0 and right_ear > 0 else 0.0
    
    blink_history.append(avg_ear)
    if len(blink_history) > max_frames:
        blink_history.pop(0)
    
    blink_count = 0
    for i in range(1, len(blink_history)):
        if blink_history[i-1] > ear_threshold and blink_history[i] < ear_threshold:
            blink_count += 1
    
    return blink_count >= min_blinks, avg_ear

# Fungsi untuk deteksi senyum
def detect_smile(face):
    if not hasattr(face, 'landmark_2d_106') or face.landmark_2d_106 is None:
        logger.warning("Landmark tidak tersedia untuk deteksi senyum")
        return False
    
    mouth = face.landmark_2d_106[60:68]
    width = np.linalg.norm(mouth[0] - mouth[4])
    height = np.linalg.norm(mouth[2] - mouth[6])
    return width / height > 1.8

# Fungsi untuk analisis gerakan wajah
def detect_motion(face, prev_landmarks, motion_threshold=2.0):
    if not hasattr(face, 'landmark_2d_106') or face.landmark_2d_106 is None or prev_landmarks is None:
        return False, getattr(face, 'landmark_2d_106', None)
    
    current_landmarks = face.landmark_2d_106
    motion = np.mean(np.linalg.norm(current_landmarks - prev_landmarks, axis=1))
    return motion > motion_threshold, current_landmarks

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
def save_attendance(student_id, status="Present", ear=0.0, motion=0.0, is_smiling=False):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log = pd.DataFrame([[student_id, timestamp, status, ear, motion, is_smiling]], 
                          columns=["Student_ID", "Timestamp", "Status", "EAR", "Motion", "Is_Smiling"])
        if os.path.exists(OUTPUT_PATH):
            log.to_csv(OUTPUT_PATH, mode='a', header=False, index=False)
        else:
            log.to_csv(OUTPUT_PATH, mode='w', header=True, index=False)
        logger.info(f"Absensi tersimpan: {student_id} - {status} (EAR: {ear:.2f}, Motion: {motion:.2f}, Smiling: {is_smiling})")
    except Exception as e:
        logger.error(f"Error menyimpan absensi: {e}")

# Fungsi untuk buat grafik frekuensi absensi
def plot_attendance_frequency(csv_path, plot_path):
    try:
        df = pd.read_csv(csv_path)
        attendance_counts = df['Student_ID'].value_counts()
        
        plt.figure(figsize=(8, 6))
        attendance_counts.plot(kind='bar', color='skyblue')
        plt.title('Frekuensi Absensi Mahasiswa')
        plt.xlabel('Student ID')
        plt.ylabel('Jumlah Absensi')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        plt.figure(figsize=(8, 6))
        attendance_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Distribusi Absensi')
        plt.savefig(plot_path.replace('.png', '_pie.png'))
        plt.close()
        
        logger.info(f"Grafik disimpan di: {plot_path}")
    except Exception as e:
        logger.error(f"Error membuat grafik: {e}")

# Fungsi untuk validasi akurasi
def validate_accuracy(database, test_images_path, liveness_threshold=0.6):
    true_labels = []
    pred_labels = []
    liveness_results = []
    
    for student_id in os.listdir(test_images_path):
        if student_id.startswith('.'):
            continue
        student_path = os.path.join(test_images_path, student_id)
        if os.path.isdir(student_path):
            for img_name in os.listdir(student_path):
                if img_name.startswith('.'):
                    continue
                img_path = os.path.join(student_path, img_name)
                img = cv2.imread(img_path)
                img = preprocess_image(img)
                if img is None:
                    continue
                
                faces = app.get(img)
                if len(faces) == 0:
                    continue
                
                face = faces[0]
                if not hasattr(face, 'landmark_2d_106') or face.landmark_2d_106 is None:
                    continue
                
                blink_history = []
                is_blinking, ear = detect_blink(face, blink_history)
                is_smiling = detect_smile(face)
                identity, distance = verify_face(face.embedding, database)
                
                true_labels.append(student_id)
                pred_labels.append(identity if identity and distance < liveness_threshold else None)
                liveness_results.append(is_blinking or is_smiling)
    
    accuracy = accuracy_score(true_labels, pred_labels, normalize=True)
    liveness_accuracy = sum(liveness_results) / len(liveness_results) if liveness_results else 0.0
    return accuracy, liveness_accuracy

# Fungsi untuk gambar progress bar
def draw_progress_bar(frame, progress, x, y, width=200, height=20):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), -1)  # Background
    fill_width = int(width * progress)
    cv2.rectangle(frame, (x, y), (x + fill_width, y + height), (0, 255, 0), -1)  # Fill
    cv2.putText(frame, f"Progress: {int(progress * 100)}%", (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# Fungsi utama
def main():
    # Muat database
    logger.info("Memuat database wajah...")
    database = load_face_database()
    if not database:
        logger.warning("Database kosong, tetapi melanjutkan untuk tes webcam.")
    
    # Validasi akurasi (opsional)
    test_images_path = os.path.join(BASE_DIR, "../assets/test_images")
    if os.path.exists(test_images_path) and database:
        try:
            accuracy, liveness_accuracy = validate_accuracy(database, test_images_path)
            logger.info(f"Akurasi Face Recognition: {accuracy:.2f}")
            logger.info(f"Akurasi Liveness Detection: {liveness_accuracy:.2f}")
        except Exception as e:
            logger.error(f"Error validasi akurasi: {e}")
    
    # Buka webcam
    for cam_idx in range(2):
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            logger.info(f"Webcam terbuka dengan index: {cam_idx}")
            break
        cap.release()
    else:
        logger.error("Gagal membuka webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Variabel untuk verifikasi dan anti-spoofing
    last_verified_id = None
    no_face_frames = 0
    verification_timeout = time.time()
    VERIFICATION_INTERVAL = 5
    blink_history = []
    prev_landmarks = None
    liveness_confirmed = False
    liveness_timeout = time.time()
    LIVENESS_CHECK_INTERVAL = 10
    step = 1  # Langkah liveness (1: kedip, 2: senyum, 3: gerakan)
    blink_detected = False
    smile_detected = False
    motion_detected = False
    progress = 0.0

    logger.info("Sistem absensi anti-spoofing dimulai. Arahkan wajah ke kamera. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Gagal membaca frame dari webcam!")
            continue
        
        frame = preprocess_image(frame)
        if frame is None:
            continue
        
        try:
            faces = app.get(frame)
        except Exception as e:
            logger.error(f"Error deteksi wajah: {e}")
            faces = []
        
        if len(faces) == 0:
            no_face_frames += 1
            if no_face_frames >= 30:
                last_verified_id = None
                verification_timeout = time.time()
                liveness_confirmed = False
                blink_history = []
                prev_landmarks = None
                step = 1
                blink_detected = False
                smile_detected = False
                motion_detected = False
                progress = 0.0
                logger.info("Reset verifikasi: Tidak ada wajah terdeteksi")
            cv2.putText(frame, "Wajah tidak terdeteksi, coba lebih dekat", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            no_face_frames = 0
            face = faces[0]
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            current_time = time.time()
            if current_time - liveness_timeout >= LIVENESS_CHECK_INTERVAL or not liveness_confirmed:
                is_blinking, ear = detect_blink(face, blink_history)
                has_motion, prev_landmarks = detect_motion(face, prev_landmarks)
                is_smiling = detect_smile(face)
                
                # Langkah demi langkah untuk liveness
                if step == 1:
                    if is_blinking:
                        blink_detected = True
                        progress = 0.33
                        cv2.putText(frame, "Kedip terdeteksi! Lanjutkan.", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        step = 2
                    else:
                        cv2.putText(frame, "Langkah 1: Kedip 2 kali", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                elif step == 2:
                    if is_smiling:
                        smile_detected = True
                        progress = 0.66
                        cv2.putText(frame, "Senyum terdeteksi! Lanjutkan.", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        step = 3
                    else:
                        cv2.putText(frame, "Langkah 2: Tersenyum", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                elif step == 3:
                    if has_motion:
                        motion_detected = True
                        progress = 1.0
                        cv2.putText(frame, "Gerakan terdeteksi! Wajah hidup.", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        liveness_confirmed = True
                        liveness_timeout = current_time
                    else:
                        cv2.putText(frame, "Langkah 3: Gerakkan kepala sedikit", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Jika liveness gagal setelah timeout
                if not liveness_confirmed and current_time - liveness_timeout >= LIVENESS_CHECK_INTERVAL:
                    cv2.putText(frame, "Coba kedip atau tersenyum lagi!", (50, 130), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    logger.info(f"Liveness gagal: Blinking={is_blinking}, Smiling={is_smiling}, Motion={has_motion}, EAR={ear:.2f}")
            
            # Gambar progress bar
            draw_progress_bar(frame, progress, 50, 160)
            
            # Verifikasi wajah
            if liveness_confirmed and database and (current_time - verification_timeout >= VERIFICATION_INTERVAL or last_verified_id is None):
                identity, distance = verify_face(face.embedding, database)
                if identity:
                    cv2.putText(frame, f"Absensi: {identity} (Dist: {distance:.2f})", 
                               (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if identity != last_verified_id:
                        motion = np.mean(np.linalg.norm(face.landmark_2d_106 - prev_landmarks, axis=1)) if prev_landmarks is not None else 0.0
                        save_attendance(identity, ear=ear, motion=motion, is_smiling=is_smiling)
                        last_verified_id = identity
                        verification_timeout = current_time
                        logger.info(f"Mahasiswa terverifikasi: {identity}, Distance: {distance:.2f}")
                else:
                    cv2.putText(frame, f"Tidak dikenali (Dist: {distance:.2f})", 
                               (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    logger.info(f"Wajah tidak dikenali, Distance: {distance:.2f}")
            else:
                if liveness_confirmed and last_verified_id:
                    cv2.putText(frame, f"Absensi: {last_verified_id} (Terakhir)", 
                               (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif not database:
                    cv2.putText(frame, "Database kosong!", (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, "Menunggu verifikasi wajah...", 
                               (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Fraudless Attendance System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if os.path.exists(OUTPUT_PATH):
        plot_attendance_frequency(OUTPUT_PATH, PLOT_PATH)
    
    logger.info("Sistem absensi selesai.")

if __name__ == "__main__":
    main()