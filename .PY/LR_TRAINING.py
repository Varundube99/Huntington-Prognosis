
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# --- 1. Load Data ---
PROCESSED_CLEAN_FILE = "../Dataset/pre_processed_dataset.csv"
RAW_DATA_FILE = "../Dataset/hd_dataset.csv"

print(f"Loading preprocessed data from '{PROCESSED_CLEAN_FILE}'...")
df_processed = pd.read_csv(PROCESSED_CLEAN_FILE)

target_column = 'Disease_Stage'
X = df_processed.drop(target_column, axis=1)
y = df_processed[target_column]

raw_df = pd.read_csv(RAW_DATA_FILE)
target_le = LabelEncoder()
target_le.fit(raw_df['Disease_Stage'])
class_names = target_le.classes_

# --- 2. Split Data ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nData split into:")
print(f"{len(X_train)} training samples (80%)")
print(f"{len(X_val)} validation samples (10%)")
print(f"{len(X_test)} testing samples (10%)")

# --- 3. Create and Train the Model Pipeline ---
model = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

print("\nTraining the Logistic Regression model...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- 4. Evaluate on the VALIDATION set ---
y_pred_val = model.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)

print("\n--- VALIDATION SET RESULTS ---")
print(f"--- MODEL ACCURACY ON VALIDATION SET: {accuracy_val:.4f} ---")

cm_val = confusion_matrix(y_val, y_pred_val)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Logistic Regression Confusion Matrix (Validation Set)', fontsize=14)
plt.ylabel('Actual Stage')
plt.xlabel('Predicted Stage')
plt.show()

# --- 5. FULL Evaluation on the TEST set ---
print("\n--- FULL TEST SET RESULTS ---")
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)

accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"--- FINAL MODEL ACCURACY ON TEST SET: {accuracy_test:.4f} ---")

cm_test = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
plt.title('Logistic Regression Confusion Matrix (Test Set)', fontsize=14)
plt.ylabel('Actual Stage')
plt.xlabel('Predicted Stage')
plt.show()

# --- NEW: Classification Report (Precision, Recall, F1-Score) ---
print("\nClassification Report (Test Set):")
report = classification_report(y_test, y_pred_test, target_names=class_names)
print(report)
print("-" * 50)

# --- NEW: Multi-Class ROC/AUC Curve ---
print("Generating ROC/AUC Curve for the Test Set...")
y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_binarized.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba_test[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class "{class_names[i]}" (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression Multi-Class ROC Curve')
plt.legend(loc="lower right")
plt.show()