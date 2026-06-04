import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Memuat Pengklasifikasi Kaskade Haar (Pastikan file XML berada di folder yang sama atau jalurnya benar)
# Jika kamu pakai Google Colab, sesuaikan foldernya seperti di PDF: '/content/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# 2. Fungsi untuk Mendeteksi Wajah
def adjusted_detect_face(img):
    face_img = img.copy()
    # Deteksi wajah (mengembalikan koordinat kotak persegi panjang)
    face_rect = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
    
    # Menggambar kotak di setiap wajah yang terdeteksi
    for (x, y, w, h) in face_rect:
        # Format: cv2.rectangle(gambar, pt1, pt2, warna_bgr, ketebalan)
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
        
    return face_img

# 3. Fungsi untuk Mendeteksi Mata
def detect_eyes(img):
    eye_img = img.copy()
    # Deteksi mata
    eye_rect = eye_cascade.detectMultiScale(eye_img, scaleFactor=1.2, minNeighbors=5)
    
    # Menggambar kotak di setiap mata yang terdeteksi
    for (x, y, w, h) in eye_rect:
        cv2.rectangle(eye_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
        
    return eye_img

# 4. Memuat Gambar Sumber
# Ganti 'andrew.jpg' dengan nama file gambar yang ingin kamu uji
img = cv2.imread('saya2.jpg')

# Pastikan gambar berhasil dimuat sebelum diproses
if img is None:
    print("Error: Gambar tidak ditemukan! Periksa kembali nama file atau jalurnya.")
else:
    # Buat salinan gambar untuk masing-masing pengujian
    img_copy1 = img.copy()
    img_copy2 = img.copy()
    img_copy3 = img.copy()

    # Tampilkan gambar asli (Konversi BGR ke RGB karena OpenCV membaca dalam format BGR)
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Gambar Asli")
    plt.axis('off')
    plt.show()

    # 5. Menjalankan Deteksi Wajah
    face = adjusted_detect_face(img_copy1)
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    plt.title("Deteksi Wajah")
    plt.axis('off')
    plt.show()
    cv2.imwrite('face.jpg', face) # Menyimpan hasil

    # 6. Menjalankan Deteksi Mata
    eyes = detect_eyes(img_copy2)
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(eyes, cv2.COLOR_BGR2RGB))
    plt.title("Deteksi Mata")
    plt.axis('off')
    plt.show()
    cv2.imwrite('eyes.jpg', eyes) # Menyimpan hasil

    # 7. Menjalankan Deteksi Wajah dan Mata Sekaligus
    # Agar wajah dan mata terdeteksi bersamaan, kita masukkan hasil deteksi wajah ke deteksi mata
    face_and_eyes = detect_eyes(face)
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(face_and_eyes, cv2.COLOR_BGR2RGB))
    plt.title("Deteksi Wajah dan Mata")
    plt.axis('off')
    plt.show()
    cv2.imwrite('face+eyes.jpg', face_and_eyes) # Menyimpan hasil