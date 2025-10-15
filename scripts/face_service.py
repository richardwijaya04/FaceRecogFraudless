# app_revised.py

import os
import cv2
import numpy as np
import logging
import io
from PIL import Image
from threading import Lock

from flask import Flask, request, jsonify
from insightface.app import FaceAnalysis

# =================================================================================
# Professor's Note: Setup & Konfigurasi
# =================================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "../assets/database")
os.makedirs(DATABASE_PATH, exist_ok=True)


# =================================================================================
# Professor's Note: Class Layanan AI yang Telah Dioptimalkan
# Model tetap di-load sekali. Logika pendaftaran sekarang lebih efisien.
# =================================================================================
class FaceRecognitionService:
    def __init__(self):
        logger.info("Menginisialisasi Face Recognition Service...")
        try:
            self.face_analyzer = FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace berhasil diinisialisasi.")
        except Exception as e:
            logger.error(f"Gagal inisialisasi InsightFace: {e}")
            raise e
        
        self.db_lock = Lock()
        self.face_database = self._load_face_database()

    def _load_face_database(self):
        """
        Memuat semua wajah dari direktori database ke memori.
        Setiap entri akan menyimpan embedding rata-rata dan jumlah foto.
        Format: { student_id: {'embedding': np.array, 'count': int} }
        """
        database = {}
        logger.info("Memuat database wajah dari disk...")
        for student_id in os.listdir(DATABASE_PATH):
            student_path = os.path.join(DATABASE_PATH, student_id)
            if os.path.isdir(student_path):
                embeddings = []
                image_files = [f for f in os.listdir(student_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if not image_files:
                    continue

                for file in image_files:
                    try:
                        img_path = os.path.join(student_path, file)
                        img = cv2.imread(img_path)
                        if img is None: continue
                        
                        faces = self.face_analyzer.get(img)
                        if faces:
                            embeddings.append(faces[0].embedding)
                    except Exception as e:
                        logger.warning(f"Gagal memproses {img_path}: {e}")
                
                if embeddings:
                    # Simpan embedding rata-rata dan jumlah foto
                    database[student_id] = {
                        'embedding': np.mean(embeddings, axis=0),
                        'count': len(embeddings)
                    }
                    logger.info(f"Berhasil memuat data: {student_id} ({len(embeddings)} gambar)")
        logger.info("Database wajah selesai dimuat.")
        return database

    def identify_face(self, embedding_to_check, threshold=0.6):
        """Mencari wajah yang paling cocok dari seluruh database (1:N)."""
        if not self.face_database:
            return "Unknown", float('inf')

        min_dist = float('inf')
        identity = "Unknown"
        
        with self.db_lock:
            # Iterasi melalui database yang sekarang berisi dict
            for student_id, data in self.face_database.items():
                db_embedding = data['embedding']
                dist = 1 - np.dot(embedding_to_check, db_embedding) / (np.linalg.norm(embedding_to_check) * np.linalg.norm(db_embedding))
                if dist < min_dist:
                    min_dist = dist
                    identity = student_id
        
        if min_dist > threshold:
            identity = "Unknown"
            
        return identity, min_dist

    def verify_face(self, embedding_to_check, student_id, threshold=0.6):
        """Memverifikasi apakah wajah cocok dengan student_id tertentu (1:1)."""
        with self.db_lock:
            student_data = self.face_database.get(student_id)
        
        if student_data is None:
            return False, float('inf')
            
        db_embedding = student_data['embedding']
        dist = 1 - np.dot(embedding_to_check, db_embedding) / (np.linalg.norm(embedding_to_check) * np.linalg.norm(db_embedding))
        
        return dist <= threshold, dist

    def register_face(self, image_bytes, student_id):
        """
        Mendaftarkan wajah baru. Menangani mahasiswa baru dan lama secara efisien.
        Ini adalah versi yang telah disempurnakan.
        """
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        faces = self.face_analyzer.get(img_cv)
        if not faces:
            return False, "Tidak ada wajah terdeteksi."

        new_embedding = faces[0].embedding
        
        student_path = os.path.join(DATABASE_PATH, student_id)
        os.makedirs(student_path, exist_ok=True)
        
        # Simpan gambar baru dengan nama file yang unik
        file_count = len(os.listdir(student_path)) + 1
        new_image_path = os.path.join(student_path, f"{student_id}_{file_count}.jpg")
        cv2.imwrite(new_image_path, img_cv)

        # Update database di memori secara thread-safe dan efisien
        with self.db_lock:
            if student_id not in self.face_database:
                # KASUS 1: MAHASISWA BARU
                self.face_database[student_id] = {
                    'embedding': new_embedding,
                    'count': 1
                }
                message = f"Mahasiswa baru {student_id} berhasil didaftarkan dengan foto pertama."
            else:
                # KASUS 2: MAHASISWA LAMA (OPTIMIZED UPDATE)
                existing_data = self.face_database[student_id]
                old_embedding = existing_data['embedding']
                old_count = existing_data['count']
                
                # Hitung rata-rata baru secara inkremental
                new_avg_embedding = ((old_embedding * old_count) + new_embedding) / (old_count + 1)
                
                # Update data di memori
                self.face_database[student_id]['embedding'] = new_avg_embedding
                self.face_database[student_id]['count'] += 1
                message = f"Foto baru berhasil ditambahkan untuk {student_id}. Total foto sekarang: {old_count + 1}."
        
        return True, message

# Inisialisasi service sebagai global singleton
face_service = FaceRecognitionService()


# =================================================================================
# Professor's Note: Definisi API Endpoints
# Endpoint '/register' sekarang memanggil method yang sudah dioptimalkan.
# =================================================================================

@app.route('/recognition/register', methods=['POST'])
def register():
    """Endpoint untuk mendaftarkan wajah, baik baru maupun lama."""
    if 'photo' not in request.files or 'studentId' not in request.form:
        return jsonify({"success": False, "error": "Membutuhkan 'photo' dan 'studentId'."}), 400
    
    student_id = request.form['studentId']
    photo_file = request.files['photo'].read()

    # Menggunakan fungsi baru yang lebih efisien
    success, message = face_service.register_face(photo_file, student_id)
    
    if success:
        return jsonify({"success": True, "message": message})
    else:
        return jsonify({"success": False, "error": message}), 400

# Endpoint /verify dan /identify tidak perlu diubah karena sudah membaca struktur baru
@app.route('/recognition/verify', methods=['POST'])
def verify():
    if 'photo' not in request.files or 'studentId' not in request.form:
        return jsonify({"success": False, "error": "Membutuhkan 'photo' dan 'studentId'."}), 400

    student_id = request.form['studentId']
    photo_file = request.files['photo'].read()
    
    img = Image.open(io.BytesIO(photo_file)).convert("RGB")
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    faces = face_service.face_analyzer.get(img_cv)
    if not faces:
        return jsonify({"success": False, "error": "Tidak ada wajah terdeteksi."})

    is_match, distance = face_service.verify_face(faces[0].embedding, student_id)
    similarity = (1 - distance) * 100 if distance != float('inf') else 0

    return jsonify({
        "success": True,
        "isMatch": bool(is_match),
        "studentId": student_id,
        "similarity": f"{similarity:.2f}%"
    })

@app.route('/recognition/identify', methods=['POST'])
def identify():
    if 'photo' not in request.files:
        return jsonify({"success": False, "error": "Membutuhkan 'photo'."}), 400

    photo_file = request.files['photo'].read()
    
    img = Image.open(io.BytesIO(photo_file)).convert("RGB")
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    faces = face_service.face_analyzer.get(img_cv)
    if not faces:
        return jsonify({"success": False, "error": "Tidak ada wajah terdeteksi."})

    identity, distance = face_service.identify_face(faces[0].embedding)
    similarity = (1 - distance) * 100 if distance != float('inf') else 0

    return jsonify({
        "success": True,
        "identity": identity,
        "similarity": f"{similarity:.2f}%"
    })


# =================================================================================
# Professor's Note: Menjalankan Server
# =================================================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)