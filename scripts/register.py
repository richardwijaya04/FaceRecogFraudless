import cv2
import os
import logging
from insightface.app import FaceAnalysis
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inisialisasi InsightFace
try:
    app = FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition', 'landmark_2d_106'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    logger.info("InsightFace berhasil diinisialisasi")
except Exception as e:
    logger.error(f"Gagal inisialisasi InsightFace: {e}")
    exit(1)

# Path untuk database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "../assets/database")

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

# Fungsi utama untuk registrasi
def main():
    # Input student ID
    student_id = input("Masukkan Student ID (contoh: student_001): ").strip()
    if not student_id:
        logger.error("Student ID tidak boleh kosong!")
        return
    
    # Buat folder untuk student_id
    student_path = os.path.join(DATABASE_PATH, student_id)
    os.makedirs(student_path, exist_ok=True)
    
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
    
    # Variabel untuk registrasi
    image_count = 0
    max_images = 3
    instructions = [
        "Langkah 1: Lihat kamera dengan wajah netral, tekan 's' untuk simpan.",
        "Langkah 2: Tersenyum, tekan 's' untuk simpan.",
        "Langkah 3: Miringkan kepala sedikit, tekan 's' untuk simpan."
    ]
    
    logger.info(f"Memulai registrasi untuk {student_id}. Tekan 's' untuk simpan gambar, 'q' untuk keluar.")

    while image_count < max_images:
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
            cv2.putText(frame, "Wajah tidak terdeteksi, coba lebih dekat", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            face = faces[0]
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            if hasattr(face, 'embedding') and face.embedding is not None:
                cv2.putText(frame, "Wajah valid, tekan 's' untuk simpan", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Wajah tidak valid, coba lagi", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Tampilkan instruksi
        cv2.putText(frame, instructions[image_count], (50, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Gambar {image_count + 1} dari {max_images}", (50, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Registrasi Wajah", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and len(faces) > 0 and hasattr(faces[0], 'embedding') and faces[0].embedding is not None:
            img_name = f"{student_id[-3:]}_{image_count + 1}.jpeg"
            img_path = os.path.join(student_path, img_name)
            cv2.imwrite(img_path, frame)
            logger.info(f"Gambar disimpan: {img_path}")
            image_count += 1
            time.sleep(0.5)  # Jeda untuk feedback
    
    cap.release()
    cv2.destroyAllWindows()
    
    if image_count == max_images:
        logger.info(f"Registrasi selesai untuk {student_id}. {max_images} gambar disimpan.")
    else:
        logger.warning(f"Registrasi tidak lengkap, hanya {image_count} gambar disimpan.")
    
    logger.info("Sistem registrasi selesai.")

if __name__ == "__main__":
    main()