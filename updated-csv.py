import os
import pandas as pd

# Cek apakah folder UTS ada
if not os.path.exists('UTS'):
    os.makedirs('UTS')  # Buat folder UTS jika belum ada

# Load dataset dari file CSV
file_path = 'UTS/kankerpentil.csv'
data = pd.read_csv(file_path)

# Ubah kolom 'diagnosis' dari 'M' menjadi 1 dan 'B' menjadi 0
data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Ubah kolom 'diagnosis' dari 'x>0' menjadi 1 dan 'x<0' menjadi 0
# data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x > 0 else 0)

# Simpan dataset yang sudah diperbarui
updated_file_path = 'UTS/kankerpentil-ready.csv'
try:
    data.to_csv(updated_file_path, index=False)
    print(f"Data berhasil disimpan di {updated_file_path}")
except Exception as e:
    print(f"Gagal menyimpan data: {e}")

# Output beberapa baris untuk melihat hasil perubahan
print(data.head())
