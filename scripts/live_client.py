import cv2
import requests
import numpy as np

# URL dari AI Service kita
API_URL_IDENTIFY = "http://127.0.0.1:5050/identify"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Encode frame ke format JPEG dalam memory
    _, img_encoded = cv2.imencode('.jpg', frame)
    image_bytes = img_encoded.tobytes()

    try:
        # Kirim frame ke API
        response = requests.post(API_URL_IDENTIFY, files={'photo': ('frame.jpg', image_bytes, 'image/jpeg')})
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                identity = data['identity']
                similarity = data['similarity']
                
                # Gambar Bounding Box (dummy, karena API tidak mengembalikan bbox)
                # Untuk bbox presisi, API perlu di-upgrade untuk mengembalikannya
                label = f"{identity} ({similarity})"
                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print("Error dari server:", response.text)

    except requests.exceptions.RequestException as e:
        print(f"Tidak bisa terhubung ke server: {e}")

    cv2.imshow('Live Face Recognition Client', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()