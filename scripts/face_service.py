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

# Inisialisasi Flask App
app = Flask(__name__)

# Path folder (dibuat lebih robust)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "../assets/database")
os.makedirs(DATABASE_PATH, exist_ok=True)


# =================================================================================
# Professor's Note: Class untuk Enkapsulasi Layanan AI
# Ini adalah "best practice" untuk memastikan model hanya di-load sekali
# dan state (database wajah) dikelola dengan aman (thread-safe).
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
        
        # Kunci (Lock) untuk operasi database yang aman dari konflik thread
        self.db_lock = Lock()
        self.face_database = self._load_face_database()

    def _load_face_database(self):
        """Memuat semua wajah dari direktori database ke memori."""
        database = {}
        logger.info("Memuat database wajah dari disk...")
        for student_id in os.listdir(DATABASE_PATH):
            student_path = os.path.join(DATABASE_PATH, student_id)
            if os.path.isdir(student_path):
                embeddings = []
                for file in os.listdir(student_path):
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
                    database[student_id] = np.mean(embeddings, axis=0)
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
            for student_id, db_embedding in self.face_database.items():
                # Cosine Distance
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
            db_embedding = self.face_database.get(student_id)
        
        if db_embedding is None:
            return False, float('inf') # ID Siswa tidak ditemukan di database
            
        dist = 1 - np.dot(embedding_to_check, db_embedding) / (np.linalg.norm(embedding_to_check) * np.linalg.norm(db_embedding))
        
        return dist <= threshold, dist

    def register_new_face(self, image_bytes, student_id):
        """Mendaftarkan wajah baru ke database dan menyimpannya ke disk."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        faces = self.face_analyzer.get(img_cv)
        if not faces:
            return False, "Tidak ada wajah terdeteksi."

        new_embedding = faces[0].embedding
        
        student_path = os.path.join(DATABASE_PATH, student_id)
        os.makedirs(student_path, exist_ok=True)
        
        # Simpan gambar baru
        file_count = len(os.listdir(student_path)) + 1
        new_image_path = os.path.join(student_path, f"{student_id}_{file_count}.jpg")
        cv2.imwrite(new_image_path, img_cv)

        # Update database di memori secara thread-safe
        with self.db_lock:
            if student_id not in self.face_database:
                self.face_database[student_id] = new_embedding
            else:
                # Perbarui embedding rata-rata
                existing_embedding = self.face_database[student_id]
                # Kita bisa menggunakan pendekatan yang lebih canggih di sini (e.g., weighted average)
                # Tapi untuk sekarang, kita muat ulang untuk kesederhanaan
                self._update_student_embedding(student_id)
        
        return True, f"Wajah {student_id} berhasil didaftarkan."
    
    def _update_student_embedding(self, student_id):
        """Helper untuk mengupdate embedding seorang siswa di memori."""
        student_path = os.path.join(DATABASE_PATH, student_id)
        embeddings = []
        for file in os.listdir(student_path):
            img_path = os.path.join(student_path, file)
            img = cv2.imread(img_path)
            faces = self.face_analyzer.get(img)
            if faces:
                embeddings.append(faces[0].embedding)
        if embeddings:
            self.face_database[student_id] = np.mean(embeddings, axis=0)

# Inisialisasi service sebagai global singleton
face_service = FaceRecognitionService()


# =================================================================================
# Professor's Note: Definisi API Endpoints
# Ini adalah 'pintu' bagi aplikasi lain untuk berinteraksi dengan AI kita.
# =================================================================================

@app.route('/recognition/register', methods=['POST'])
def register():
    """Endpoint untuk mendaftarkan wajah baru."""
    if 'photo' not in request.files or 'studentId' not in request.form:
        return jsonify({"success": False, "error": "Membutuhkan 'photo' dan 'studentId'."}), 400
    
    student_id = request.form['studentId']
    photo_file = request.files['photo'].read()

    success, message = face_service.register_new_face(photo_file, student_id)
    
    if success:
        return jsonify({"success": True, "message": message})
    else:
        return jsonify({"success": False, "error": message}), 400


@app.route('/recognition/verify', methods=['POST'])
def verify():
    """Endpoint untuk memverifikasi wajah dengan ID yang diklaim (1:1)."""
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
    similarity = (1 - distance) * 100

    return jsonify({
        "success": True,
        "isMatch": bool(is_match),
        "studentId": student_id,
        "similarity": f"{similarity:.2f}%"
    })

@app.route('/recognition/identify', methods=['POST'])
def identify():
    """Endpoint untuk mengidentifikasi wajah dari database (1:N)."""
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
    similarity = (1 - distance) * 100

    return jsonify({
        "success": True,
        "identity": identity,
        "similarity": f"{similarity:.2f}%"
    })


# =================================================================================
# Professor's Note: Menjalankan Server
# =================================================================================
if __name__ == '__main__':
    # host='0.0.0.0' agar bisa diakses dari luar container/mesin
    app.run(host='0.0.0.0', port=5050, debug=False)