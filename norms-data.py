import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# # Load dataset
# file_path = 'UTS/kankerpentil-ready.csv'  # Ganti dengan path file Anda
# data = pd.read_csv(file_path)

# # Langkah 1: Memilih hanya kolom numerik
# numeric_data = data.select_dtypes(include=[np.number])

# # Fungsi untuk mengecek missing value
# def check_missing_values(df):
#     missing_values = df.isnull().sum()
#     print("Jumlah Missing Value di setiap kolom:")
#     print(missing_values)
#     return missing_values

# # Pengecekan missing value pada data sebelum preprocessing
# print("Pengecekan Missing Value Sebelum Preprocessing:")
# missing_values_before = check_missing_values(data)

# # Langkah 2: Deteksi Outliers Sebelum Preprocessing
# def detect_outliers(df):
#     Q1 = df.quantile(0.25)
#     Q3 = df.quantile(0.75)
#     IQR = Q3 - Q1
#     outliers_condition = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
#     return outliers_condition

# # Visualisasi Outliers Sebelum Pembersihan
# def plot_outliers(df, title):
#     plt.figure(figsize=(12, 6))
#     sns.boxplot(data=df)
#     plt.title(title)
#     plt.xticks(rotation=90)
#     plt.show()

# # Tampilkan data outliers sebelum preprocessing
# print("\nOutliers Sebelum Preprocessing:")
# outliers_before = detect_outliers(numeric_data)
# print(outliers_before.sum())  # Jumlah outliers di setiap kolom
# plot_outliers(numeric_data, "Outliers Sebelum Preprocessing")

# # Langkah 3: Preprocessing - Membersihkan Outliers
# def clean_outliers(df):
#     Q1 = df.quantile(0.25)
#     Q3 = df.quantile(0.75)
#     IQR = Q3 - Q1
#     outliers_condition = (df < (Q1 - 1.5 * IQR)) | (Q3 + 1.5 * IQR < df)
    
#     # Ganti outlier dengan nilai median kolom
#     for col in df.columns:
#         if outliers_condition[col].any():  # Hanya jika ada outliers
#             median_value = df[col].median()  # Nilai median kolom
#             df[col] = np.where(outliers_condition[col], median_value, df[col])  # Ganti dengan median
#     return df

# # Pembersihan outliers
# data_cleaned = clean_outliers(numeric_data.copy())

# # Langkah 4: Deteksi Outliers Setelah Preprocessing
# print("\nOutliers Setelah Preprocessing:")
# outliers_after = detect_outliers(data_cleaned)
# print(outliers_after.sum())  # Jumlah outliers di setiap kolom
# plot_outliers(data_cleaned, "Outliers Setelah Preprocessing")

# # Langkah 5: Simpan data yang sudah dibersihkan
# # Gabungkan data numerik yang sudah dibersihkan dengan kolom non-numerik asli
# data_final = data.copy()
# data_final[numeric_data.columns] = data_cleaned

# cleaned_file_path = 'UTS/kankerpentil-preprocessing.csv'  # Ganti dengan path tujuan penyimpanan
# data_final.to_csv(cleaned_file_path, index=False)

# print(f"\nData yang sudah dibersihkan disimpan di: {cleaned_file_path}")

# # Pengecekan missing value pada data setelah preprocessing
# print("\nPengecekan Missing Value Setelah Preprocessing:")
# missing_values_after = check_missing_values(data_final)



# Load dataset
file_path = 'UTS/kankerpentil-ready.csv'  # Ganti dengan path file Anda
data = pd.read_csv(file_path)

# Langkah 1: Memilih hanya kolom numerik kecuali 'id'
numeric_data = data.select_dtypes(include=[np.number]).drop(columns=['id'], errors='ignore')

# Fungsi untuk mengecek missing value
def check_missing_values(df):
    missing_values = df.isnull().sum()
    print("Jumlah Missing Value di setiap kolom:")
    print(missing_values)
    return missing_values

# Pengecekan missing value pada data sebelum preprocessing
print("Pengecekan Missing Value Sebelum Preprocessing:")
missing_values_before = check_missing_values(data)

# Langkah 2: Deteksi Outliers Sebelum Preprocessing (kecuali 'id')
def detect_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers_condition = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
    return outliers_condition

# Visualisasi Outliers Sebelum Pembersihan
def plot_outliers(df, title):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.show()

# Tampilkan data outliers sebelum preprocessing
print("\nOutliers Sebelum Preprocessing:")
outliers_before = detect_outliers(numeric_data)
print(outliers_before.sum())  # Jumlah outliers di setiap kolom
plot_outliers(numeric_data, "Outliers Sebelum Preprocessing")

# Langkah 3: Preprocessing - Membersihkan Outliers
def clean_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers_condition = (df < (Q1 - 1.5 * IQR)) | (Q3 + 1.5 * IQR < df)
    
    # Ganti outlier dengan nilai median kolom
    for col in df.columns:
        if outliers_condition[col].any():  # Hanya jika ada outliers
            median_value = df[col].median()  # Nilai median kolom
            df[col] = np.where(outliers_condition[col], median_value, df[col])  # Ganti dengan median
    return df

# Pembersihan outliers
data_cleaned = clean_outliers(numeric_data.copy())

# Langkah 4: Deteksi Outliers Setelah Preprocessing
print("\nOutliers Setelah Preprocessing:")
outliers_after = detect_outliers(data_cleaned)
print(outliers_after.sum())  # Jumlah outliers di setiap kolom
plot_outliers(data_cleaned, "Outliers Setelah Preprocessing")

# Langkah 5: Simpan data yang sudah dibersihkan
# Gabungkan data numerik yang sudah dibersihkan dengan kolom non-numerik asli
data_final = data.copy()
data_final[numeric_data.columns] = data_cleaned

cleaned_file_path = 'UTS/kankerpentil-preprocessing-not.id.csv'  # Ganti dengan path tujuan penyimpanan
data_final.to_csv(cleaned_file_path, index=False)

print(f"\nData yang sudah dibersihkan disimpan di: {cleaned_file_path}")

# Pengecekan missing value pada data setelah preprocessing
print("\nPengecekan Missing Value Setelah Preprocessing:")
missing_values_after = check_missing_values(data_final)
