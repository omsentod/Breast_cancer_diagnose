import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load dataset
file_path = 'UTS/kankerpentil-minmax.csv'  # Ganti dengan path file Anda
data = pd.read_csv(file_path)

# Pisahkan fitur dan label (kolom diagnosis sebagai target)
X = data.iloc[:, 2:].values  # Menggunakan fitur dari kolom 2 ke atas (semua fitur numerik)
y = data['diagnosis'].values  # Kolom diagnosis sebagai target

# Pemodelan Random Forest
random_forest = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=3)  # Menggunakan 100 pohon dengan kedalaman maksimal 3
random_forest.fit(X, y)

# Fungsi untuk memproses input pasien
def process_input(responses):
    # Konversi input ke dalam format numerik sesuai dengan urutan fitur dalam dataset
    response_numeric = [float(resp) for resp in responses]
    return response_numeric

# Fungsi untuk mensimulasikan input jika tidak ingin memasukkan secara manual
def get_input():
    responses = [
0.6071749727862179,0.4206966520121743,0.5957432105590492,0.47359490986214203,0.41238602509704797,0.255567143120054,0.34653233364573566,0.4720675944333996,0.2636363636363638,0.08403538331929239,0.234184320115879,0.14515558698727016,0.24068227865994443,0.19723271286033942,0.1625250705374443,0.12525911016312677,0.08563131313131313,0.288122750520932,0.07989531153261663,0.038078852452220056,0.6897901102810388,0.5026652452025587,0.6792668957617412,0.5438458513566653,0.5284950141979792,0.2791376817921627,0.42907348242811505,0.820618556701031,0.2371377882909521,0.1384625475534566    ]
    return responses

# Pertanyaan untuk input manual
questions = [
    "Masukkan nilai radius_mean: ",
    "Masukkan nilai texture_mean: ",
    "Masukkan nilai perimeter_mean: ",
    "Masukkan nilai area_mean: ",
    "Masukkan nilai smoothness_mean: ",
    "Masukkan nilai compactness_mean: ",
    "Masukkan nilai concavity_mean: ",
    "Masukkan nilai concave points_mean: ",
    "Masukkan nilai symmetry_mean: ",
    "Masukkan nilai fractal_dimension_mean: ",
    "Masukkan nilai radius_se: ",
    "Masukkan nilai texture_se: ",
    "Masukkan nilai perimeter_se: ",
    "Masukkan nilai area_se: ",
    "Masukkan nilai smoothness_se: ",
    "Masukkan nilai compactness_se: ",
    "Masukkan nilai concavity_se: ",
    "Masukkan nilai concave points_se: ",
    "Masukkan nilai symmetry_se: ",
    "Masukkan nilai fractal_dimension_se: ",
    "Masukkan nilai radius_worst: ",
    "Masukkan nilai texture_worst: ",
    "Masukkan nilai perimeter_worst: ",
    "Masukkan nilai area_worst: ",
    "Masukkan nilai smoothness_worst: ",
    "Masukkan nilai compactness_worst: ",
    "Masukkan nilai concavity_worst: ",
    "Masukkan nilai concave points_worst: ",
    "Masukkan nilai symmetry_worst: ",
    "Masukkan nilai fractal_dimension_worst: "
]

# Input dari pengguna atau menggunakan simulasi
use_simulation = input("Apakah Anda ingin menggunakan input simulasi? (Y/N): ").strip().upper()
if use_simulation == "Y":
    responses = get_input()
else:
    responses = []
    print("Input Data Pasien")
    for q in questions:
        responses.append(input(q))

# Memproses input pasien
input_pasien = process_input(responses)
print(f"Data input pasien: {input_pasien}")

# Membuat prediksi
predtest = random_forest.predict([input_pasien])
hasil_prediksi = "Malignant" if predtest == 1 else "Benign"
print(f"Hasil prediksi: Pasien {hasil_prediksi}")
print(predtest)
