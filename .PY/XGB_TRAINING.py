import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV

# --- 1. Load Pre-processed Dataset ---
PROCESSED_CLEAN_FILE = "../Dataset/pre_processed_dataset.csv"

print("Loading data for XGBoost hyperparameter tuning...")
df_processed = pd.read_csv(PROCESSED_CLEAN_FILE)
print("Data loaded successfully.")

# --- 2. Prepare Data ---
target_column = 'Disease_Stage'
X = df_processed.drop(target_column, axis=1)
y = df_processed[target_column]

# We only need a training set to perform the search.
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nUsing {len(X_train)} samples for the search.")


# --- 3. Define the Model and Parameter Grid ---
# Create a base XGBoost model instance.
# The `use_label_encoder=False` and `eval_metric='mlogloss'` are often recommended
# for modern versions of XGBoost to avoid warnings and set a clear objective.
xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Define the grid of hyperparameters to test.
# This is a focused grid to keep the search time manageable.
param_grid = {
    'max_depth': [3, 5],               # Maximum depth of a tree
    'n_estimators': [100, 200],          # Number of boosting rounds
    'learning_rate': [0.1, 0.01],      # Step size shrinkage
    'subsample': [0.8, 1.0],               # Subsample ratio of the training instance
    'colsample_bytree': [0.8, 1.0]         # Subsample ratio of columns when constructing each tree
}

print("\nParameter grid for testing:")
print(param_grid)


# --- 4. Set Up and Run GridSearchCV ---
grid_search = GridSearchCV(
    estimator=xgb_classifier,
    param_grid=param_grid,
    cv=3,  # Using 3-fold CV to reduce runtime; 5 is also a good choice
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

print("\nStarting GridSearchCV for XGBoost... This will likely take several minutes.")
grid_search.fit(X_train, y_train)


# --- 5. Display the Best Parameters ---
print("\n--------------------------------------------------")
print("GridSearchCV has completed.")
print(f"The best cross-validation accuracy score was: {grid_search.best_score_:.4f}")
print("The best parameters found for the XGBoost model are:")
print(grid_search.best_params_)
print("--------------------------------------------------")
print("\nYou can now use these parameters to train your final XGBoost model.")


# Actual codee

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from itertools import cycle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# --- 1. Load Your Datasets ---
PROCESSED_CLEAN_FILE = "../Dataset/pre_processed_dataset.csv"
RAW_DATA_FILE = "../Dataset/hd_dataset.csv"

print("Loading data from your files...")
df_processed = pd.read_csv(PROCESSED_CLEAN_FILE)
raw_df = pd.read_csv(RAW_DATA_FILE)
print("Files loaded successfully.")

# --- 2. Prepare Data ---
target_column = 'Disease_Stage'
X = df_processed.drop(target_column, axis=1)
y = df_processed[target_column]

target_le = LabelEncoder()
target_le.fit(raw_df['Disease_Stage'])
class_names = target_le.classes_

# --- 3. Split Data (80% Train, 10% Validation, 10% Test) ---
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

# --- 4. Create and Train the Model with Your Best Parameters ---
print("\nTraining the final XGBoost model with the best parameters...")

# Initialize the model with the exact parameters from your GridSearchCV results
model = xgb.XGBClassifier(
    objective='multi:softmax',
    use_label_encoder=False,
    eval_metric='mlogloss',
    colsample_bytree=0.8,
    learning_rate=0.1,
    max_depth=3,
    n_estimators=100,
    subsample=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Evaluate on the VALIDATION Set ---
y_pred_val = model.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
print("\n--- VALIDATION SET RESULTS ---")
print(f"--- Accuracy on Validation Set: {accuracy_val:.4f} ---")
cm_val = confusion_matrix(y_val, y_pred_val)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
plt.title('Tuned XGBoost Confusion Matrix (Validation Set)', fontsize=14)
plt.ylabel('Actual Stage')
plt.xlabel('Predicted Stage')
plt.show()

# --- 6. Comprehensive Evaluation on the TEST Set ---
print("\n--- COMPREHENSIVE TEST SET RESULTS ---")
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)

accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"Accuracy: {accuracy_test:.4f}")
print("-" * 50)

print("Confusion Matrix:")
cm_test = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Tuned XGBoost Confusion Matrix (Test Set)', fontsize=14)
plt.ylabel('Actual Stage')
plt.xlabel('Predicted Stage')
plt.show()

print("\nClassification Report:")
report = classification_report(y_test, y_pred_test, target_names=class_names)
print(report)
print("-" * 50)

print("Generating ROC/AUC Curve...")
y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_binarized.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba_test[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
colors = cycle(['purple', 'lime', 'cyan', 'magenta', 'yellow'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Tuned XGBoost Multi-Class ROC Curve')
plt.legend(loc="lower right")
plt.show()