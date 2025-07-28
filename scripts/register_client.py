# scripts/register_client.py

import requests
import sys

# Cek apakah argumen sudah benar
if len(sys.argv) != 3:
    print("Cara penggunaan: python scripts/register_client.py <student_id> <path_ke_gambar>")
    sys.exit(1)

student_id = sys.argv[1]
image_path = sys.argv[2]

API_URL_REGISTER = "http://127.0.0.1:5050/register"

try:
    with open(image_path, 'rb') as f:
        files = {'photo': (image_path, f, 'image/jpeg')}
        data = {'studentId': student_id}
        
        response = requests.post(API_URL_REGISTER, files=files, data=data)
        
        print("Status Kode:", response.status_code)
        print("Respon Server:", response.json())

except FileNotFoundError:
    print(f"Error: File tidak ditemukan di '{image_path}'")
except requests.exceptions.RequestException as e:
    print(f"Error: Tidak bisa terhubung ke server. Pastikan server berjalan.\n{e}")