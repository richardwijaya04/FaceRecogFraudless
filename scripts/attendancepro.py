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

# Inisialisasi model
try:
    app = FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("InsightFace berhasil diinisialisasi")
except Exception as e:
    logger.error(f"Gagal inisialisasi InsightFace: {e}")
    exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "../assets/database")
TEST_PATH = os.path.join(BASE_DIR, "../assets/test_data")  # Folder berisi test images

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
    return database

def verify_face(embedding, database, threshold=0.6):
    identity = None
    min_dist = float('inf')
    for student_id, db_embedding in database.items():
        dist = 1 - (embedding @ db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
        if dist < min_dist:
            min_dist = dist
            identity = student_id
    if min_dist < threshold:
        return identity
    return "Unknown"

def evaluate_model(database):
    y_true = []
    y_pred = []
    
    for student_id in os.listdir(TEST_PATH):
        student_folder = os.path.join(TEST_PATH, student_id)
        if not os.path.isdir(student_folder):
            continue
        for img_file in os.listdir(student_folder):
            img_path = os.path.join(student_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Gagal baca gambar {img_path}")
                continue
            faces = app.get(img)
            if not faces:
                logger.warning(f"Tidak ada wajah di {img_path}")
                continue
            prediction = verify_face(faces[0].embedding, database)
            y_true.append(student_id)
            y_pred.append(prediction)

    # Evaluasi
    y_true_bin = [1 if y != "Unknown" else 0 for y in y_true]
    y_pred_bin = [1 if y != "Unknown" else 0 for y in y_pred]

    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    accuracy = np.mean([yt == yp for yt, yp in zip(y_true, y_pred)])

    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1 Score : {f1:.2f}")
    print(f"Accuracy : {accuracy:.2f}")

    return y_true, y_pred

def main():
    logger.info("Muat database wajah...")
    database = load_face_database()
    if not database:
        logger.error("Database kosong.")
        return
    logger.info("Evaluasi dimulai...")
    y_true, y_pred = evaluate_model(database)

if __name__ == "__main__":
    main()
