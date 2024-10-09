import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load dataset
file_path = 'UTS/kankerpentil-prepo.csv'  # Ganti dengan path file Anda
data = pd.read_csv(file_path)

# Pisahkan fitur dan label (kolom diagnosis sebagai target)
X = data.iloc[:, 2:].values 
y = data['diagnosis'].values  

# Splitting Data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Pemodelan Decision Tree dengan pembatasan kedalaman (max_depth)
print("PEMODELAN DENGAN DECISION TREE (TERBATAS MAX DEPTH)".center(75,"="))
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=3)  # Batasi kedalaman tree hingga 3
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_test)

# Menghitung Akurasi
accuracy_dt = round(accuracy_score(y_test, Y_pred) * 100, 2)
print(f'Akurasi Model Decision Tree: {accuracy_dt}%')

# Menghitung Confusion Matrix
cm = confusion_matrix(y_test, Y_pred)
print('CLASSIFICATION REPORT DECISION TREE'.center(75, '='))

# Menampilkan Precision, Recall, F1-Score, dan Support
print(classification_report(y_test, Y_pred))

# Menghitung nilai True Positives, False Positives, True Negatives, dan False Negatives
TN = cm[1][1] * 1.0  # True Negative
FN = cm[1][0] * 1.0  # False Negative
TP = cm[0][0] * 1.0  # True Positive
FP = cm[0][1] * 1.0  # False Positive

# Menghitung Sensitivitas dan Spesifisitas
sens = TN / (TN + FP) * 100
spec = TP / (TP + FN) * 100

# Menampilkan hasil
print('Akurasi : ', accuracy_dt, "%")
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
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# Visualisasi Decision Tree dengan ukuran plot yang terbatas dan font lebih kecil
plt.figure(figsize=(10, 6))  # Membatasi ukuran visualisasi tree
plot_tree(decision_tree, filled=True, feature_names=data.columns[2:], class_names=['Benign', 'Malignant'], rounded=True, fontsize=8)  
plt.title("Decision Tree (Max Depth = 10)")
plt.show()
