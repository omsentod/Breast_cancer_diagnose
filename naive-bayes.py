import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load dataset
file_path = 'UTS/kankerpentilsiap.csv'  # Ganti dengan path file Anda
data = pd.read_csv(file_path)

# Pisahkan fitur dan label (kolom diagnosis sebagai target)
X = data.iloc[:, 2:].values  # Menggunakan fitur dari kolom 2 ke atas
y = data['diagnosis'].values  # Kolom diagnosis sebagai target
print("data variabel".center(75,"="))
print(X)
print("data kelas".center(75,"="))
print(y)

print("data awal".center(75,"="))
print(data)

#pengecekan missing value
print("pengecekan missing value".center(75,"="))
print(data.isnull().sum())
print("============================================================")

# Splitting Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#pembagian training dan testing
print("SPLITTING DATA 20-80".center(75,"="))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print("instance variabel data training".center(75,"="))
print(X_train)
print("instance kelas data training".center(75,"="))
print(y_train)
print("instance variabel data testing".center(75,"="))
print(X_test)
print("instance kelas data testing".center(75,"="))
print(y_test)

# Pemodelan Naive Bayes 
print("PEMODELAN DENGAN NAIVE BAYES".center(75,"="))
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test) 

# Menghitung Akurasi
accuracy_nb = round(accuracy_score(y_test, Y_pred) * 100, 2)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

print("Instance Prediksi Naive Bayes:")
print(Y_pred)

# Perhitungan Confusion Matrix
cm = confusion_matrix(y_test, Y_pred)
print('CLASSIFICATION REPORT NAIVE BAYES'.center(75, '='))

# Mendapat Akurasi
accuracy = accuracy_score(y_test, Y_pred)
precision = precision_score(y_test, Y_pred)

# Menampilkan Precision, Recall, F1-Score, dan Support
print(classification_report(y_test, Y_pred))

# Menghitung nilai True Positives, False Positives, True Negatives, dan False Negatives
TN = cm[1][1] * 1.0  # True Negative
FN = cm[1][0] * 1.0  # False Negative
TP = cm[0][0] * 1.0  # True Positive
FP = cm[0][1] * 1.0  # False Positive
total = TN + FN + TP + FP

# Menghitung Sensitivitas dan Spesifisitas
sens = TN / (TN + FP) * 100
spec = TP / (TP + FN) * 100

# Menampilkan hasil
print('Akurasi : ',accuracy * 100,"%")
print('Sensitivity : ' + str(sens))
print('Specificity : ' + str(spec))
print('Precision : ' + str(precision))

# Visualisasi Confusion Matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot()

# Plot Confusion Matrix sebagai Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt=".0f")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes")
plt.show()
