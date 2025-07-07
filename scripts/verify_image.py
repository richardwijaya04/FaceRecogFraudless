import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from insightface.app import FaceAnalysis
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inisialisasi model InsightFace
try:
    app = FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("InsightFace berhasil diinisialisasi")
except Exception as e:
    logger.error(f"Gagal inisialisasi InsightFace: {e}")
    exit(1)

# Path folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "../assets/database")

# Muat database wajah dari folder
def load_face_database():
    database = {}
    for student_id in os.listdir(DATABASE_PATH):
        path = os.path.join(DATABASE_PATH, student_id)
        if os.path.isdir(path):
            embeddings = []
            for file in os.listdir(path):
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                faces = app.get(img)
                if faces:
                    embeddings.append(faces[0].embedding)
            if embeddings:
                database[student_id] = np.mean(embeddings, axis=0)
                logger.info(f"Berhasil load data: {student_id} ({len(embeddings)} gambar)")
            else:
                logger.warning(f"Tidak ada embedding valid untuk: {student_id}")
    return database

# Verifikasi wajah: cari identitas terdekat
def verify_face(embedding, database, threshold=0.6):
    identity = None
    min_dist = float('inf')
    for student_id, db_embedding in database.items():
        dist = 1 - (embedding @ db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
        if dist < min_dist:
            min_dist = dist
            identity = student_id
    if min_dist < threshold:
        return identity, min_dist
    return "Unknown", min_dist

# Fungsi prediksi dari input gambar
def predict_image(image_path, database, threshold=0.6):
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Gagal membaca gambar dari path: {image_path}")
        return

    faces = app.get(img)
    if not faces:
        logger.warning("❌ Tidak ada wajah terdeteksi dalam gambar.")
        return

    face = faces[0]
    identity, dist = verify_face(face.embedding, database, threshold)

    if identity == "Unknown":
        print("❌ Wajah tidak dikenali (Unknown)")
        print(f"🔎 Distance: {dist:.4f} (semakin kecil semakin cocok)")
    else:
        similarity = (1 - dist) * 100
        print(f"✅ Wajah dikenali sebagai: {identity}")
        print(f"🔎 Kemiripan (similarity): {similarity:.2f}% (Distance: {dist:.4f})")

    # Tampilkan gambar dengan bounding box
    bbox = face.bbox.astype(int)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    label = identity if identity != "Unknown" else "Unknown"
    cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Hasil Verifikasi", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Fungsi utama
def main():
    logger.info("Memuat database wajah...")
    database = load_face_database()
    if not database:
        logger.error("Database kosong! Pastikan folder '../assets/database/' berisi folder student.")
        return

    image_path = input("Masukkan path gambar (misal: ./test.jpg): ").strip()
    if not os.path.isfile(image_path):
        print("❌ Gambar tidak ditemukan! Pastikan path benar.")
        return

    predict_image(image_path, database)

# Eksekusi
if __name__ == "__main__":
    main()
