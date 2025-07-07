import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import Optional
from insightface.app import FaceAnalysis
from io import BytesIO
from PIL import Image
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Load InsightFace model
face_model = FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
face_model.prepare(ctx_id=0, det_size=(640, 640))

# Load Face Database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "assets/database")

def load_face_database():
    database = {}
    for student_id in os.listdir(DATABASE_PATH):
        student_path = os.path.join(DATABASE_PATH, student_id)
        if os.path.isdir(student_path):
            embeddings = []
            for img_name in os.listdir(student_path):
                img_path = os.path.join(student_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                faces = face_model.get(img)
                if faces:
                    embeddings.append(faces[0].embedding)
            if embeddings:
                database[student_id] = np.mean(embeddings, axis=0)
    return database

face_database = load_face_database()

def verify_face(embedding, database, threshold=0.6):
    min_dist = float('inf')
    identity = None
    for student_id, db_embedding in database.items():
        dist = 1 - (embedding @ db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
        if dist < min_dist:
            min_dist = dist
            identity = student_id
    if min_dist < threshold:
        return identity, min_dist
    return None, min_dist

# API Request/Response Schema
class VerificationResult(BaseModel):
    verified: bool
    identity: Optional[str] = None
    distance: Optional[float] = None

@app.post("/verify", response_model=VerificationResult)
async def verify_image(file: UploadFile = File(...)):
    contents = await file.read()
    img = np.array(Image.open(BytesIO(contents)).convert("RGB"))
    
    faces = face_model.get(img)
    if not faces:
        return VerificationResult(verified=False)

    embedding = faces[0].embedding
    identity, distance = verify_face(embedding, face_database)

    if identity:
        return VerificationResult(verified=True, identity=identity, distance=round(distance, 4))
    else:
        return VerificationResult(verified=False, distance=round(distance, 4))
