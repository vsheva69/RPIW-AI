import requests
from dotenv import load_dotenv
load_dotenv()
import os 

# Konfigurasi API Login Eksternal
API_LOGIN_URL = "https://api-splp.layanan.go.id/t/pu.go.id/ehrm/login/1.0.0"
API_KEY =  os.getenv("API_EHRM_KEY")  # Tambahkan API key Anda

def authenticate_user(username, password):
    """
    Autentikasi pengguna melalui API eksternal.
    Args:
        username (str): Username pengguna.
        password (str): Password pengguna.
    Returns:
        dict: Informasi pengguna jika berhasil login, None jika gagal.
    """
    headers = {"Apikey": f"{API_KEY}"}
    response = requests.post(API_LOGIN_URL, json={"uname": username, "pass": password}, headers=headers)
    if response.status_code == 200:
        try:
             data = response.json()  # Konversi respons ke JSON
             if data.get("status") == True:
                return {"user_id": data["nip"], "username": data["nama"]}
        except ValueError:
             print("Gagal mengonversi respons ke JSON.")
    else:
        print(f"Login gagal dengan status code: {response.status_code}")
        print("Response:", response.text)
    return None
