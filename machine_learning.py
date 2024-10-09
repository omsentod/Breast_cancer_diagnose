import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler 


# Load dataset
file_path = 'UTS/kankerpentil-prepo.csv'
data = pd.read_csv(file_path)

# Assume target column is 'diagnosis' and features start from the 3rd column
X = data.iloc[:, 2:].values  # Features
y = data['diagnosis'].values  # Target

# Transform features using Min-Max scaling
scaler = MinMaxScaler()  # Menggunakan MinMaxScaler
X = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train models
# Random Forest Model
random_forest = RandomForestClassifier(random_state=0)
random_forest.fit(X_train, y_train)

# Naive Bayes Model
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# Decision Tree Model
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=3)
decision_tree.fit(X_train, y_train)

# Function to calculate specificity
def calculate_specificity(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return specificity * 100

# Evaluate models
def evaluate_model_with_specificity(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted') * 100
    recall = recall_score(y_test, y_pred, average='weighted') * 100
    f1 = f1_score(y_test, y_pred, average='weighted') * 100
    specificity = calculate_specificity(y_test, y_pred)

    print(f"========== {model_name} Model Evaluation ==========")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    print(f"Specificity: {specificity:.2f}%\n")
    
    return accuracy, precision, recall, f1, specificity

# Evaluate all models
rf_accuracy, rf_precision, rf_recall, rf_f1, rf_specificity = evaluate_model_with_specificity(random_forest, X_test, y_test, "Random Forest")
nb_accuracy, nb_precision, nb_recall, nb_f1, nb_specificity = evaluate_model_with_specificity(naive_bayes, X_test, y_test, "Naive Bayes")
dt_accuracy, dt_precision, dt_recall, dt_f1, dt_specificity = evaluate_model_with_specificity(decision_tree, X_test, y_test, "Decision Tree")

# Aggregate the scores
rf_avg_score = np.mean([rf_accuracy, rf_precision, rf_recall, rf_f1, rf_specificity])
nb_avg_score = np.mean([nb_accuracy, nb_precision, nb_recall, nb_f1, nb_specificity])
dt_avg_score = np.mean([dt_accuracy, dt_precision, dt_recall, dt_f1, dt_specificity])

# Compare model scores
print("\n========== Model Score Comparison (Average of All Metrics) ==========")
print(f"Random Forest Avg Score: {rf_avg_score:.2f}%")
print(f"Naive Bayes Avg Score: {nb_avg_score:.2f}%")
print(f"Decision Tree Avg Score: {dt_avg_score:.2f}%")

# Find the best model based on average score
best_model_name = max(
    [("Random Forest", rf_avg_score), ("Naive Bayes", nb_avg_score), ("Decision Tree", dt_avg_score)],
    key=lambda x: x[1]
)[0]

print(f"\nAlgoritma dengan skor rata-rata terbaik adalah: {best_model_name}")

# Visualize the results using matplotlib
labels = ['Random Forest', 'Naive Bayes', 'Decision Tree']
accuracies = [rf_accuracy, nb_accuracy, dt_accuracy]
precisions = [rf_precision, nb_precision, dt_precision]
recalls = [rf_recall, nb_recall, dt_recall]
f1_scores = [rf_f1, nb_f1, dt_f1]
specificities = [rf_specificity, nb_specificity, dt_specificity]

x = np.arange(len(labels))
width = 0.15

fig, ax = plt.subplots(figsize=(12, 7))
bars1 = ax.bar(x - 2*width, accuracies, width, label='Accuracy')
bars2 = ax.bar(x - width, precisions, width, label='Precision')
bars3 = ax.bar(x, recalls, width, label='Recall')
bars4 = ax.bar(x + width, f1_scores, width, label='F1 Score')
bars5 = ax.bar(x + 2*width, specificities, width, label='Specificity')

ax.set_xlabel('Models')
ax.set_ylabel('Percentage (%)')
ax.set_title('Model Evaluation Metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def add_bar_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_bar_labels(bars1)
add_bar_labels(bars2)
add_bar_labels(bars3)
add_bar_labels(bars4)
add_bar_labels(bars5)

plt.show()
