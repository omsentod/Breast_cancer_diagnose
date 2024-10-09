import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier  # Menggunakan RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load dataset
file_path = 'UTS/kankerpentilsiap.csv'  # Ganti dengan path file Anda
data = pd.read_csv(file_path)

# Pisahkan fitur dan label (kolom diagnosis sebagai target)
X = data.iloc[:, 2:].values  # Menggunakan fitur dari kolom 2 ke atas
y = data['diagnosis'].values  # Kolom diagnosis sebagai target

# Splitting Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Pemodelan Random Forest
print("PEMODELAN DENGAN RANDOM FOREST".center(75,"="))
random_forest = RandomForestClassifier(random_state=0, n_estimators=100)  # Menggunakan 100 pohon
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)

# Menghitung Akurasi
accuracy_rf = round(accuracy_score(y_test, Y_pred) * 100, 2)
print(f'Akurasi Model Random Forest: {accuracy_rf}%')

# Menghitung Confusion Matrix
cm = confusion_matrix(y_test, Y_pred)
print('CLASSIFICATION REPORT RANDOM FOREST'.center(75, '='))

# Menampilkan Precision, Recall, F1-Score, dan Support
print(classification_report(y_test, Y_pred))

# Menghitung nilai True Positives, False Positives, True Negatives, dan False Negatives
TN = cm[1][1] * 1.0  
FN = cm[1][0] * 1.0  
TP = cm[0][0] * 1.0  
FP = cm[0][1] * 1.0  

# Menghitung Sensitivitas dan Spesifisitas
sens = TN / (TN + FP) * 100
spec = TP / (TP + FN) * 100

# Menampilkan hasil
print('Akurasi : ', accuracy_rf, "%")
print('Sensitivity : ' + str(sens))
print('Specificity : ' + str(spec))

# Visualisasi Confusion Matrix (batasi ukuran output)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot(cmap='Blues')

# Plot Confusion Matrix sebagai Heatmap (dengan ukuran yang lebih kecil)
plt.figure(figsize=(4, 4))  # Batasi ukuran heatmap
sns.heatmap(cm, annot=True, fmt=".0f", cmap='Blues', cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Visualisasi Feature Importance
plt.figure(figsize=(12, 6))
feature_importances = random_forest.feature_importances_
features = data.columns[2:]  # Mengambil nama-nama fitur dari kolom dataset

# Sort feature importance in descending order
sorted_idx = np.argsort(feature_importances)[::-1]
plt.barh(features[sorted_idx], feature_importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Feature Importance - Random Forest")
plt.show()
