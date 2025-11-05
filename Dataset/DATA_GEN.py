# --- Realistic Huntington's Disease Dataset Generator ---
# This version adds user-specified columns like Chorea_Score and
# genetic modifier details to simulate a complex, messy dataset.

import pandas as pd
import numpy as np
import uuid

def generate_realistic_hd_dataset(n_samples=48768):
    """
    Generates a synthetic dataset with both useful and user-specified unnecessary columns.
    """
    print(f"Generating a new, messy dataset with {n_samples} records...")
    
    np.random.seed(42)
    
    # --- Core Useful Features ---
    patient_ids = [f'HD_{i+1:04d}' for i in range(n_samples)]
    sex = np.random.choice(['Male', 'Female'], n_samples, p=[0.5, 0.5])
    family_history = np.random.choice(['Yes', 'No'], n_samples, p=[0.75, 0.25])
    cag_repeats = np.random.normal(loc=48, scale=6, size=n_samples).astype(int)
    cag_repeats = np.clip(cag_repeats, 30, 70)
    noise_onset = np.random.normal(0, 5, n_samples)
    age_of_onset = (70 - (cag_repeats - 40) * 1.5 + noise_onset).astype(int)
    disease_duration = np.random.gamma(2, 5, n_samples)
    age = (age_of_onset + disease_duration).astype(int)
    age = np.clip(age, 25, 80)
    age_of_onset = np.minimum(age, age_of_onset)
    duration_effect = (age - age_of_onset)
    motor_score = np.clip(( (cag_repeats - 35) * 1.5 + duration_effect * 1.2 + np.random.normal(0, 15, n_samples) ).astype(int), 0, 100)
    cognitive_score = np.clip(( 100 - (cag_repeats - 35) * 1.2 - duration_effect * 1.1 + np.random.normal(0, 15, n_samples) ).astype(int), 0, 100)
    functional_capacity = np.clip(( 100 - (motor_score * 0.4) - ((100 - cognitive_score) * 0.4) + np.random.normal(0, 10, n_samples) ).astype(int), 0, 100)

    # --- NEW: User-Requested Columns ---
    # 1. Chorea_Score: Correlated with motor score but with its own noise.
    chorea_score = np.clip(motor_score * 0.3 + np.random.normal(5, 5, n_samples), 0, 10).round(2)
    
    # 2. Genetic Modifier Details: These are high-cardinality and should be dropped.
    # We define realistic profiles to make the data plausible.
    modifier_profiles = [
        {'Gene/Factor': 'HTT', 'Function': 'CAG Trinucleotide Repeat Expansion', 'Effect': 'Neurodegeneration', 'Category': 'Primary Cause'},
        {'Gene/Factor': 'MSH3', 'Function': 'Mismatch Repair', 'Effect': 'CAG Repeat Expansion', 'Category': 'Trans-acting Modifier'},
        {'Gene/Factor': 'HTT (Somatic Expansion)', 'Function': 'CAG Repeat Instability', 'Effect': 'Faster Disease Onset', 'Category': 'Cis-acting Modifier'},
        {'Gene/Factor': 'FAN1', 'Function': 'DNA Repair', 'Effect': 'Delayed Onset', 'Category': 'Trans-acting Modifier'}
    ]
    # Assign profiles to patients. Most will be Primary Cause.
    assigned_indices = np.random.choice(len(modifier_profiles), n_samples, p=[0.7, 0.1, 0.1, 0.1])
    modifier_data = [modifier_profiles[i] for i in assigned_indices]
    
    gene_factor = [d['Gene/Factor'] for d in modifier_data]
    function = [d['Function'] for d in modifier_data]
    effect = [d['Effect'] for d in modifier_data]
    category = [d['Category'] for d in modifier_data]
    
    # --- Probabilistic Disease Stage Assignment ---
    stages = []
    for i in range(n_samples):
        fc, cag = functional_capacity[i], cag_repeats[i]
        if cag < 36: stage = 'No Disease'
        elif fc > 70: stage = np.random.choice(['Early', 'Middle'], p=[0.9, 0.1])
        elif fc > 40: stage = np.random.choice(['Early', 'Middle', 'Severe'], p=[0.15, 0.7, 0.15])
        else: stage = np.random.choice(['Middle', 'Severe'], p=[0.1, 0.9])
        stages.append(stage)
    disease_stage = np.array(stages)

    # Assemble the full DataFrame with all new columns
    df = pd.DataFrame({
        'Patient_ID': patient_ids, 'Age': age, 'Sex': sex, 'Family_History': family_history,
        'HTT_CAG_Repeat_Length': cag_repeats, 'Age_of_Onset': age_of_onset,
        'Motor_Score': motor_score, 'Cognitive_Score': cognitive_score, 
        'Chorea_Score': chorea_score, 'Functional_Capacity_Score': functional_capacity,
        'Gene/Factor': gene_factor, 'Function': function, 'Effect': effect, 'Category': category,
        'Disease_Stage': disease_stage
    })
    
    print("Messy dataset generation complete.")
    return df

if __name__ == "__main__":
    num_records = 48768
    realistic_df = generate_realistic_hd_dataset(n_samples=num_records)
    output_filename = f'realistic_hd_dataset_{num_records}.csv'
    realistic_df.to_csv(output_filename, index=False)
    print(f"Successfully created '{output_filename}' with {len(realistic_df)} records and your specified columns.")

