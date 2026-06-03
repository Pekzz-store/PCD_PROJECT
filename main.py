# 1. IMPORT PUSTAKA (LIBRARY) YANG DIBUTUHKAN
import cv2                                     # Untuk pemrosesan citra dasar (OpenCV)
import numpy as np                             # Untuk manipulasi matriks angka
import matplotlib.pyplot as plt                # Untuk visualisasi/menampilkan gambar
from scipy.ndimage import minimum_filter       # Untuk filter reduksi noise sesuai proposal
from skimage.metrics import mean_squared_error # Untuk menghitung MSE
from skimage.metrics import peak_signal_noise_ratio # Untuk menghitung PSNR

# ==============================================================================
# 2. INPUT DATASET & PREPROCESSING
# ==============================================================================

# Membaca citra Rontgen (Ganti nama file ini saat demo: misal 'tangan_1.jpg')
image_path = 'gigi_2.jpg' 
img = cv2.imread(image_path)

if img is None:
    print("Error: Gambar tidak ditemukan. Cek lagi nama filenya ya!")
else:
    # A. Konversi Citra Warna ke Grayscale (Hitam Putih)
    # Tujuannya untuk menyederhanakan matriks piksel sebelum dideteksi tepinya
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # B. Reduksi Noise (Minimum Filter 3x3)
    # Tujuannya menghaluskan derau/bercak tanpa menghilangkan detail tepi tulang
    filtered_img = minimum_filter(gray, size=3)

    # ==============================================================================
    # 3. EKSEKUSI ALGORITMA DETEKSI TEPI (EDGE DETECTION)
    # ==============================================================================
    
    # --- METODE SOBEL ---
    # Mencari arah dan besar gradien pada sumbu horizontal (X) dan vertikal (Y)
    sobelx = cv2.Sobel(filtered_img, cv2.CV_64F, 1, 0, ksize=3) 
    sobely = cv2.Sobel(filtered_img, cv2.CV_64F, 0, 1, ksize=3) 
    
    # Menghitung Magnitudo Gradien (jarak Euclidean) untuk menggabungkan X dan Y
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_mag = np.uint8(np.absolute(sobel_mag))
    
    # --- METODE PREWITT ---
    # Mendefinisikan kernel matriks Prewitt secara manual (-1, 0, 1)
    kernelx_prewitt = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely_prewitt = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    
    # Menerapkan konvolusi 2D dengan kernel Prewitt
    prewittx = cv2.filter2D(filtered_img, cv2.CV_64F, kernelx_prewitt)
    prewitty = cv2.filter2D(filtered_img, cv2.CV_64F, kernely_prewitt)
    
    # Menghitung Magnitudo Gradien (jarak Euclidean)
    prewitt_mag = cv2.magnitude(prewittx, prewitty)
    prewitt_mag = np.uint8(np.absolute(prewitt_mag))

    # ==============================================================================
    # 4. BINERISASI (THRESHOLDING)
    # ==============================================================================
    
    # Mengubah hasil magnitudo jadi citra biner (murni hitam-putih)
    # Menggunakan nilai ambang batas (Threshold) T = 70 sesuai proposal
    # Jika piksel > 70 jadi putih (255), jika < 70 jadi hitam (0)
    _, sobel_thresh = cv2.threshold(sobel_mag, 70, 255, cv2.THRESH_BINARY)
    _, prewitt_thresh = cv2.threshold(prewitt_mag, 70, 255, cv2.THRESH_BINARY)

    # ==============================================================================
    # 5. EVALUASI KUANTITATIF (MSE, PSNR, NNZ)
    # ==============================================================================
    
    # Evaluasi hasil metode SOBEL
    mse_sobel = mean_squared_error(gray, sobel_thresh)
    psnr_sobel = peak_signal_noise_ratio(gray, sobel_thresh, data_range=255)
    nnz_sobel = np.count_nonzero(sobel_thresh) # Menghitung total piksel putih (garis tepi)
    
    # Evaluasi hasil metode PREWITT
    mse_prewitt = mean_squared_error(gray, prewitt_thresh)
    psnr_prewitt = peak_signal_noise_ratio(gray, prewitt_thresh, data_range=255)
    nnz_prewitt = np.count_nonzero(prewitt_thresh)

    # Menampilkan teks angka evaluasi di terminal / layar bawah
    print("=== HASIL EVALUASI METODE SOBEL ===")
    print(f"MSE  : {mse_sobel:.2f} (Makin tinggi makin banyak tepi terdeteksi)")
    print(f"PSNR : {psnr_sobel:.2f} dB (Makin rendah makin tepat)")
    print(f"NNZ  : {nnz_sobel} piksel putih\n")
    
    print("=== HASIL EVALUASI METODE PREWITT ===")
    print(f"MSE  : {mse_prewitt:.2f}")
    print(f"PSNR : {psnr_prewitt:.2f} dB")
    print(f"NNZ  : {nnz_prewitt} piksel putih")

    # ==============================================================================
    # 6. VISUALISASI HASIL (MENAMPILKAN GAMBAR)
    # ==============================================================================
    
    # Membuka jendela ukuran 12x6 inci untuk menampilkan 3 gambar sejajar
    plt.figure(figsize=(12, 6))
    
    # Gambar 1: Citra Grayscale Asli
    plt.subplot(1, 3, 1)
    plt.title("Citra Asli (Grayscale)")
    plt.imshow(gray, cmap='gray')
    plt.axis('off') # Menyembunyikan garis koordinat
    
    # Gambar 2: Hasil Sobel
    plt.subplot(1, 3, 2)
    plt.title(f"Metode Sobel (T=70)\nPSNR: {psnr_sobel:.2f}")
    plt.imshow(sobel_thresh, cmap='gray')
    plt.axis('off')
    
    # Gambar 3: Hasil Prewitt
    plt.subplot(1, 3, 3)
    plt.title(f"Metode Prewitt (T=70)\nPSNR: {psnr_prewitt:.2f}")
    plt.imshow(prewitt_thresh, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout() # Agar posisi gambar tidak saling tumpang tindih
    plt.show()         # Tampilkan ke layar!