import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Load dataset
file_path = 'UTS/kankerpentil-prepo.csv'  # Ganti dengan path file Anda
data = pd.read_csv(file_path)

# Langkah 1: Memilih hanya kolom numerik
numeric_data = data.select_dtypes(include=[np.number])


# Menggunakan RobustScaler
scaler = RobustScaler()
data_scaled = scaler.fit_transform(data_scaled)

# Konversi kembali hasil scaling ke dalam DataFrame
data_scaled_df = pd.DataFrame(data_scaled, columns=numeric_data.columns)

# Gabungkan dengan kolom non-numerik asli
data_final = data.copy()
data_final[numeric_data.columns] = data_scaled_df

# Langkah 3: Simpan dataset yang sudah di-scale
output_file_path = 'UTS/kankerpentil-minmax.csv'  # Ganti dengan path tujuan penyimpanan
data_final.to_csv(output_file_path, index=False)

print(f"Data yang sudah di-scale menggunakan Min-Max disimpan di: {output_file_path}")
