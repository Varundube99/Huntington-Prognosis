import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')



RAW_DATA_FILE = "../Dataset/hd_dataset.csv"

# Load the dataset
print(f"Loading raw data from '{RAW_DATA_FILE}'...")
df = pd.read_csv(RAW_DATA_FILE)

# Fill missing numerical values with the median of their respective column
for col in df.select_dtypes(include=['int64', 'float64']).columns:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)


print("Performing feature extraction...")

if 'Age' in df.columns and 'Age_of_Onset' in df.columns:
    df['Disease_Duration'] = df['Age'] - df['Age_of_Onset']
   
    df['Disease_Duration'] = df['Disease_Duration'].clip(lower=0)
    print("Created new feature: 'Disease_Duration'.")

    print("\nDataset with the new 'Disease_Duration' column:")
    display(df.head())
else:
    print("Could not create 'Disease_Duration' as 'Age' or 'Age_of_Onset' is missing.")


print("Performing Label Encoding on all text-based columns...")
print(f"Shape of DataFrame before encoding: {df.shape}")


le = LabelEncoder()


for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])
    print(f"Label encoded '{col}'.")
    
print(f"\nShape of DataFrame after encoding: {df.shape} (No new columns added)")

print("\nFirst 5 rows of the fully numerical DataFrame:")
display(df.head())


PROCESSED_CLEAN_FILE = "../Dataset/pre_processed_dataset.csv"

try:
    df.to_csv(PROCESSED_CLEAN_FILE, index=False)
    print(f"\nSuccessfully saved fully preprocessed data to '{PROCESSED_CLEAN_FILE}'")
except Exception as e:
    print(f"Error saving file: {e}")