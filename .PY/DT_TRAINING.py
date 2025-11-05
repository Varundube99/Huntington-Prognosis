import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# --- 1. Load Pre-processed Dataset ---
PROCESSED_CLEAN_FILE = "/pre_processed_dataset.csv"

print("Loading data for hyperparameter tuning...")
df_processed = pd.read_csv(PROCESSED_CLEAN_FILE)
print("Data loaded successfully.")

# --- 2. Prepare Data ---
# Define features (X) and target (y)
target_column = 'Disease_Stage'
X = df_processed.drop(target_column, axis=1)
y = df_processed[target_column]

# We only need a training set to perform the search.
# We'll use an 80% split, as in your original code.
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nUsing {len(X_train)} samples for the search.")

# --- 3. Define the Model and Parameter Grid ---
# Create a base Decision Tree model instance
dt_classifier = DecisionTreeClassifier(random_state=42)

# Define the grid of hyperparameters you want to test
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("\nParameter grid for testing:")
print(param_grid)

# --- 4. Set Up and Run GridSearchCV ---
# Set up GridSearchCV with 5-fold cross-validation
# n_jobs=-1 uses all available CPU cores to speed up the process
grid_search = GridSearchCV(
    estimator=dt_classifier,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2  # verbose=2 gives more detailed output during the search
)

print("\nStarting GridSearchCV... This may take a few moments.")
# Fit the grid search to find the best parameters using only the training data
grid_search.fit(X_train, y_train)

# --- 5. Display the Best Parameters ---
print("\n--------------------------------------------------")
print("GridSearchCV has completed.")
print(f"The best cross-validation accuracy score was: {grid_search.best_score_:.4f}")
print("The best parameters found for the Decision Tree are:")
print(grid_search.best_params_)
print("--------------------------------------------------")
print("\nYou can now use these parameters to train your final model.")



# actual code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report, # New import
    roc_curve,             # New import
    auc                    # New import
)

# --- 1. Load Your Datasets ---
# Corrected file paths for the environment
PROCESSED_CLEAN_FILE = "/pre_processed_dataset.csv"
RAW_DATA_FILE = "/content/hd_dataset.csv"


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
class_names = target_le.classes_ # Store class names for plotting

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

# --- 4. Create and Train the Model ---
print("\nTraining the final Decision Tree model...")
model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=3,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42
)
model.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Evaluate on the VALIDATION Set ---
# This section remains unchanged
y_pred_val = model.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)

print("\n--- TUNED MODEL: VALIDATION SET RESULTS ---")
print(f"--- Accuracy on Validation Set: {accuracy_val:.4f} ---")
cm_val = confusion_matrix(y_val, y_pred_val)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix(Validation Set)', fontsize=14)
plt.ylabel('Actual Stage')
plt.xlabel('Predicted Stage')
plt.show()


# --- 6. FULL Evaluation on the TEST Set ---
print("\n--- FULL EVALUATION: TEST SET RESULTS ---")
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test) # Needed for ROC curve

# --- Accuracy and Confusion Matrix (as before) ---
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"--- Final Model Accuracy on Test Set: {accuracy_test:.4f} ---")
cm_test = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Purples', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix(Test Set)', fontsize=14)
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
# Binarize the output for ROC curve calculation
y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_binarized.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba_test[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves on one graph
plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class "{class_names[i]}" (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2) # Dashed line for random chance
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()