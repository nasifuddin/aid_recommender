# # src/feature_engineering.py

# import pandas as pd
# import numpy as np
# import logging
# import json # Add import for saving feature list
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split # For splitting
# import warnings
# warnings.filterwarnings('ignore') # Suppress potential warnings

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def load_and_clean_data(filepath: str) -> pd.DataFrame:
#     """
#     Loads the dataset with correct headers and performs initial cleaning.
#     """
#     logger.info(f"Loading data from {filepath} (with headers)")
#     df = pd.read_csv(filepath)

#     logger.info(f"Initial data shape: {df.shape}")
#     logger.info(f"Columns: {list(df.columns)}")

#     # Basic info
#     logger.info(f"Data types:\n{df.dtypes}")
#     logger.info(f"Missing values:\n{df.isnull().sum()}")

#     # --- Core Cleaning & Normalization ---
#     # Convert Participant_Birthdate to datetime if possible
#     df['Participant_Birthdate'] = pd.to_datetime(df['Participant_Birthdate'], errors='coerce')

#     # Standardize booleans for categorical columns like Marrital_Status
#     # Assuming 'Married' and 'Others(Widow/Unmarried)' are the values from the sample
#     # Map to 0/1 if needed for consistency, or keep as categorical and encode later
#     # For now, keep as categorical string, but ensure consistency
#     df['Marrital_Status'] = df['Marrital_Status'].str.strip() # Remove leading/trailing spaces if any

#     # Convert binary targets
#     df['Participant_Selected_For_AID'] = df['Participant_Selected_For_AID'].map({'Yes': 1, 'No': 0}).astype('Int8')
#     df['Aid_Type_Recomended'] = df['Aid_Type_Recomended'].str.strip() # Clean up potential whitespace

#     return df


# def engineer_features(df: pd.DataFrame) -> tuple:
#     """
#     Applies the feature engineering plan.
#     Returns the engineered DataFrame and the fitted LabelEncoder.
#     """
#     logger.info("Starting feature engineering...")
#     df_fe = df.copy()

#     # --- 3.1 Household structure & vulnerability ---
#     # Assuming Family_Size is the total household size
#     df_fe['hh_size'] = df_fe['Family_Size']

#     # Dependency ratio (example: assuming 'has_under5' and 'has_50_plus' represent dependents)
#     # This is a simplification; ideally, we'd have age bands
#     df_fe['dependents_count'] = df_fe['has_under5'] + df_fe['has_50_plus']
#     df_fe['working_age_adults'] = df_fe['has_18_50_Family_member'] # Assuming this is a count or flag for working age
#     # Handle division by zero
#     df_fe['dependency_ratio'] = df_fe['dependents_count'] / (df_fe['working_age_adults'] + 1e-8)

#     # --- 3.2 Economic capacity & liquidity ---
#     df_fe['income_total_monthly'] = df_fe['Income_Monthly_per_head'] * df_fe['hh_size']
#     df_fe['expenditure_total_monthly'] = df_fe['Income_Monthly_per_head'] * df_fe['hh_size'] # Proxy
#     df_fe['income_pc'] = df_fe['Income_Monthly_per_head']
#     df_fe['expenditure_pc'] = df_fe['expenditure_total_monthly'] / df_fe['hh_size']
#     df_fe['savings_to_expenditure'] = df_fe['Savings_Amt'] / (df_fe['expenditure_total_monthly'] + 1e-8)
#     df_fe['debt_to_income'] = df_fe['Loans_Outstanding'] / (df_fe['Total_Income_Annualy'] + 1.0)
#     df_fe['debt_to_income'] = df_fe['debt_to_income'].replace([float('inf'), float('-inf')], np.nan)
#     df_fe['liquidity_gap'] = (df_fe['expenditure_total_monthly'] - df_fe['income_total_monthly']) / (df_fe['income_total_monthly'] + 1e-8)

#     # --- 3.3 Assets & livelihood readiness ---
#     df_fe['productive_asset_index'] = df_fe['Productive_Asset_Value']

#     # --- 3.4 Health burden & care access ---
#     df_fe['chronic_illness_count'] = df_fe['Chronic_Patient_Num']
#     df_fe['disability_in_household'] = df_fe['Disabled_Yes']

#     # --- 4. Aid-Type–Specific “Readiness/Need” Scores ---
#     # Standardize inputs for scoring (using StandardScaler or min-max)
#     scaler = StandardScaler()
#     # Example features for each score (using available features, might need adjustment)
#     cash_need_features = ['liquidity_gap', 'debt_to_income'] # Simplified for example
#     cash_need_features = [f for f in cash_need_features if f in df_fe.columns] # Filter existing columns
#     if cash_need_features:
#         cash_need_scaled = scaler.fit_transform(df_fe[cash_need_features].fillna(0))
#         df_fe['cash_need_score'] = cash_need_scaled.mean(axis=1) # Simple average, weights can be added later
#     else:
#         df_fe['cash_need_score'] = 0 # Default if no features available

#     asset_readiness_features = ['productive_asset_index'] # Simplified for example
#     asset_readiness_features = [f for f in asset_readiness_features if f in df_fe.columns]
#     if asset_readiness_features:
#         asset_readiness_scaled = scaler.fit_transform(df_fe[asset_readiness_features].fillna(0))
#         df_fe['asset_readiness_score'] = asset_readiness_scaled.mean(axis=1)
#     else:
#         df_fe['asset_readiness_score'] = 0

#     health_need_features = ['chronic_illness_count', 'disability_in_household'] # Simplified for example
#     health_need_features = [f for f in health_need_features if f in df_fe.columns]
#     if health_need_features:
#         health_need_scaled = scaler.fit_transform(df_fe[health_need_features].fillna(0))
#         df_fe['health_need_score'] = health_need_scaled.mean(axis=1)
#     else:
#         df_fe['health_need_score'] = 0

#     training_suitability_features = ['Marrital_Status_Encoded'] # Placeholder, will be created below
#     training_suitability_features = [f for f in training_suitability_features if f in df_fe.columns]
#     if training_suitability_features:
#         training_suitability_scaled = scaler.fit_transform(df_fe[training_suitability_features].fillna(0))
#         df_fe['training_suitability_score'] = training_suitability_scaled.mean(axis=1)
#     else:
#         df_fe['training_suitability_score'] = 0

#     # Create top need tag and softmax vector (requires all four scores to be non-zero, which they are now)
#     score_cols = ['cash_need_score', 'asset_readiness_score', 'health_need_score', 'training_suitability_score']
#     score_df = df_fe[score_cols]

#     # Argmax for top need tag
#     df_fe['top_need_tag'] = score_df.idxmax(axis=1)
#     # Softmax for probability vector
#     def softmax(x):
#         exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # Subtract max for numerical stability
#         return exp_x / np.sum(exp_x, axis=1, keepdims=True)

#     softmax_vals = softmax(score_df.values)
#     for i, col in enumerate(score_cols):
#         df_fe[f'{col}_softmax'] = softmax_vals[:, i]

#     # --- 5. Encodings & Interaction Strategy ---
#     # Categorical encoding for Marrital_Status
#     le_marital = LabelEncoder()
#     df_fe['Marrital_Status_Encoded'] = le_marital.fit_transform(df_fe['Marrital_Status'].astype(str))

#     # Interaction features (example)
#     df_fe['dependency_ratio_x_income_pc'] = df_fe['dependency_ratio'] * df_fe['income_pc']
#     df_fe['productive_asset_x_market_access'] = df_fe['productive_asset_index'] * 0 # Placeholder for market_access_proxy

#     # Missingness indicators
#     for col in ['income_pc', 'debt_to_income', 'savings_to_expenditure']:
#         if col in df_fe.columns:
#             df_fe[f'is_{col}_missing'] = df_fe[col].isna().astype(int)

#     logger.info("Feature engineering completed.")
#     return df_fe, le_marital # Return the encoder


# # --- FIX ---
# # Added 'final_features: list' as an argument to the function
# def split_data(df_fe: pd.DataFrame, final_features: list, task1_target: str, task2_target: str, test_size: float = 0.2, val_size: float = 0.1):
# # --- END FIX ---
#     """
#     Splits the data into train, validation, and test sets, stratifying by targets.
#     Ensures X and y shapes are perfectly aligned before returning.
#     """
#     logger.info("Splitting data into train/val/test sets...")
#     # Combine targets for stratification if both exist and are non-null in the same rows
#     # Use rows where both targets are present for splitting if possible
#     mask = df_fe[task1_target].notna() & df_fe[task2_target].notna()
#     df_strat = df_fe[mask].copy()

#     if df_strat.empty:
#         logger.warning("No rows found where both targets are present for stratified splitting. Using simple split.")
        
#         # --- FIX ---
#         # Select *only* final_features for X_full
#         X_full = df_fe[final_features]
#         # --- END FIX ---
        
#         y1_full = df_fe[task1_target]
#         y2_full = df_fe[task2_target]

#         # Debug: Check shapes before first split (simple split path)
#         logger.info(f"Simple split path: Initial shapes - X: {X_full.shape}, y1: {y1_full.shape}, y2: {y2_full.shape}")
#         if X_full.shape[0] != y1_full.shape[0] or X_full.shape[0] != y2_full.shape[0]:
#             logger.error("FATAL ERROR: Initial shapes do not match before simple split.")
#             logger.error(f"  X: {X_full.shape}, y1: {y1_full.shape}, y2: {y2_full.shape}")
#             raise ValueError("Initial X, y1, y2 shapes do not match for simple split.")

#         # Simple split for Task 1 target (stratify on y1)
#         # train_test_split should maintain alignment between X_out and y_out
#         X_temp, X_test, y1_temp, y1_test, y2_temp, y2_test = train_test_split(
#             X_full, y1_full, y2_full, test_size=test_size, random_state=42, stratify=y1_full
#         )
#         logger.info(f"Shapes after first split (temp/test): X_temp: {X_temp.shape}, y1_temp: {y1_temp.shape}, y2_temp: {y2_temp.shape}")
#         logger.info(f"Shapes after first split (temp/test): X_test: {X_test.shape}, y1_test: {y1_test.shape}, y2_test: {y2_test.shape}")

#         # Verify alignment after first split
#         if not (X_temp.shape[0] == y1_temp.shape[0] == y2_temp.shape[0]):
#             logger.error(f"Shape mismatch after first split (temp): X {X_temp.shape[0]}, y1 {y1_temp.shape[0]}, y2 {y2_temp.shape[0]}")
#             raise ValueError("Shape mismatch after first split (temp).")
#         if not (X_test.shape[0] == y1_test.shape[0] == y2_test.shape[0]):
#             logger.error(f"Shape mismatch after first split (test): X {X_test.shape[0]}, y1 {y1_test.shape[0]}, y2 {y2_test.shape[0]}")
#             raise ValueError("Shape mismatch after first split (test).")

#         # Split temp into train and val (stratify on y1_temp)
#         X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
#             X_temp, y1_temp, y2_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y1_temp
#         )
#         logger.info(f"Shapes after second split (train/val): X_train: {X_train.shape}, y1_train: {y1_train.shape}, y2_train: {y2_train.shape}")
#         logger.info(f"Shapes after second split (train/val): X_val: {X_val.shape}, y1_val: {y1_val.shape}, y2_val: {y2_val.shape}")

#         # Verify alignment after second split
#         if not (X_train.shape[0] == y1_train.shape[0] == y2_train.shape[0]):
#             logger.error(f"Shape mismatch after second split (train): X {X_train.shape[0]}, y1 {y1_train.shape[0]}, y2 {y2_train.shape[0]}")
#             raise ValueError("Shape mismatch after second split (train).")
#         if not (X_val.shape[0] == y1_val.shape[0] == y2_val.shape[0]):
#             logger.error(f"Shape mismatch after second split (val): X {X_val.shape[0]}, y1 {y1_val.shape[0]}, y2 {y2_val.shape[0]}")
#             raise ValueError("Shape mismatch after second split (val).")

#     else:
#         logger.info(f"Found {df_strat.shape[0]} rows with both targets. Using stratified split on these rows.")
        
#         # --- FIX ---
#         # Select *only* final_features for X_strat
#         X_strat = df_strat[final_features]
#         # --- END FIX ---

#         y_strat_df = df_strat[[task1_target, task2_target]] # Keep as DataFrame for combined strat
#         # Create combined strat column string
#         y_strat_combined = y_strat_df.apply(lambda x: f"{x[task1_target]}_{x[task2_target]}", axis=1)

#         logger.info(f"Shapes for stratified split: X_strat: {X_strat.shape}, y_strat_combined: {y_strat_combined.shape}")

#         X_temp, X_test, y_temp_df, y_test_df = train_test_split(
#             X_strat, y_strat_df, test_size=test_size, random_state=42, stratify=y_strat_combined
#         )
#         logger.info(f"Shapes after first strat split (temp/test): X_temp: {X_temp.shape}, y_temp_df: {y_temp_df.shape}, y_test_df: {y_test_df.shape}")

#         # Extract y1, y2 from the resulting DataFrames
#         y1_temp = y_temp_df[task1_target]
#         y2_temp = y_temp_df[task2_target]
#         y1_test = y_test_df[task1_target]
#         y2_test = y_test_df[task2_target]
#         logger.info(f"Shapes after extracting y from temp/test: y1_temp: {y1_temp.shape}, y2_temp: {y2_temp.shape}, y1_test: {y1_test.shape}, y2_test: {y2_test.shape}")

#         # Verify alignment after first strat split
#         if not (X_temp.shape[0] == y1_temp.shape[0] == y2_temp.shape[0]):
#             logger.error(f"Shape mismatch after first strat split (temp): X {X_temp.shape[0]}, y1 {y1_temp.shape[0]}, y2 {y2_temp.shape[0]}")
#             raise ValueError("Shape mismatch after first strat split (temp).")
#         if not (X_test.shape[0] == y1_test.shape[0] == y2_test.shape[0]):
#             logger.error(f"Shape mismatch after first strat split (test): X {X_test.shape[0]}, y1 {y1_test.shape[0]}, y2 {y2_test.shape[0]}")
#             raise ValueError("Shape mismatch after first strat split (test).")

#         # Recreate combined strat column for temp split
#         y_strat_temp_combined = y_temp_df.apply(lambda x: f"{x[task1_target]}_{x[task2_target]}", axis=1)
#         X_train, X_val, y_train_df, y_val_df = train_test_split(
#             X_temp, y_temp_df, test_size=val_size/(1-test_size), random_state=42, stratify=y_strat_temp_combined
#         )
#         logger.info(f"Shapes after second strat split (train/val): X_train: {X_train.shape}, y_train_df: {y_train_df.shape}, y_val_df: {y_val_df.shape}")

#         # Extract y1, y2 from the resulting DataFrames for train/val
#         y1_train = y_train_df[task1_target]
#         y2_train = y_train_df[task2_target]
#         y1_val = y_val_df[task1_target]
#         y2_val = y_val_df[task2_target]
#         logger.info(f"Shapes after extracting y from train/val: y1_train: {y1_train.shape}, y2_train: {y2_train.shape}, y1_val: {y1_val.shape}, y2_val: {y2_val.shape}")

#         # Verify alignment after second strat split
#         if not (X_train.shape[0] == y1_train.shape[0] == y2_train.shape[0]):
#             logger.error(f"Shape mismatch after second strat split (train): X {X_train.shape[0]}, y1 {y1_train.shape[0]}, y2 {y2_train.shape[0]}")
#             raise ValueError("Shape mismatch after second strat split (train).")
#         if not (X_val.shape[0] == y1_val.shape[0] == y2_val.shape[0]):
#             logger.error(f"Shape mismatch after second strat split (val): X {X_val.shape[0]}, y1 {y1_val.shape[0]}, y2 {y2_val.shape[0]}")
#             raise ValueError("Shape mismatch after second strat split (val).")

#     # Debug: Check shapes before returning
#     logger.info(f"Final split shapes - Train: X={X_train.shape}, y1={y1_train.shape}, y2={y2_train.shape}")
#     logger.info(f"Final split shapes - Val: X={X_val.shape}, y1={y1_val.shape}, y2={y2_val.shape}")
#     logger.info(f"Final split shapes - Test: X={X_test.shape}, y1={y1_test.shape}, y2={y2_test.shape}")

#     # Corrected the typo 'y_val' to 'y2_val'
#     return (X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test)


# def run_feature_engineering(filepath: str) -> tuple:
#     """
#     Main function to run the feature engineering pipeline.
#     Returns split datasets and feature info.
#     """
#     logger.info("Starting Feature Engineering Pipeline...")

#     df = load_and_clean_data(filepath)
#     df_fe, le_marital = engineer_features(df) # Get the encoder

#     task1_target = 'Participant_Selected_For_AID'
#     task2_target = 'Aid_Type_Recomended'

#     # Identify final features (exclude targets and potentially ID columns if present)
#     exclude_cols = [task1_target, task2_target, 'Sl', 'Participant_ID', 'Participant_Birthdate', 
#                     'Marrital_Status', 'top_need_tag'] # Add other non-numeric/original string cols
#     final_features = [col for col in df_fe.columns if col not in exclude_cols]
    
#     # --- FIX ---
#     # Ensure all final features are numeric, handling potential NaNs
#     df_fe[final_features] = df_fe[final_features].apply(pd.to_numeric, errors='coerce').fillna(0)
#     # --- END FIX ---
    
#     logger.info(f"Final feature list ({len(final_features)}): {final_features[:10]}...") # Print first 10 as example

#     # --- FIX ---
#     # Pass final_features to split_data
#     X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test = \
#         split_data(df_fe, final_features, task1_target, task2_target)
#     # --- END FIX ---

#     # Debug: Check shapes before saving
#     logger.info(f"Shapes before saving - Train: X={X_train.shape}, y1={y1_train.shape}, y2={y2_train.shape}")
#     logger.info(f"Shapes before saving - Val: X={X_val.shape}, y1={y1_val.shape}, y2={y2_val.shape}")
#     logger.info(f"Shapes before saving - Test: X={X_test.shape}, y1={y1_test.shape}, y2={y2_test.shape}")

#     # Verify shapes align before saving
#     assert X_train.shape[0] == y1_train.shape[0] == y2_train.shape[0], f"Shape mismatch in Train split: X {X_train.shape[0]}, y1 {y1_train.shape[0]}, y2 {y2_train.shape[0]}"
#     assert X_val.shape[0] == y1_val.shape[0] == y2_val.shape[0], f"Shape mismatch in Val split: X {X_val.shape[0]}, y1 {y1_val.shape[0]}, y2 {y2_val.shape[0]}"
#     assert X_test.shape[0] == y1_test.shape[0] == y2_test.shape[0], f"Shape mismatch in Test split: X {X_test.shape[0]}, y1 {y1_test.shape[0]}, y2 {y2_test.shape[0]}"

#     # Save the processed datasets
#     processed_data_dict = {
#         'X_train': X_train,
#         'X_val': X_val,
#         'X_test': X_test,
#         'y1_train': y1_train,
#         'y1_val': y1_val,
#         'y1_test': y1_test,
#         'y2_train': y2_train,
#         'y2_val': y2_val,
#         'y2_test': y2_test
#     }
    
#     # Add index=False to all .to_csv() calls
#     for name, data in processed_data_dict.items():
#         if name.startswith('y'):
#              # For targets (Series), save without header and without index
#             data.to_csv(f'data/processed_{name}.csv', index=False, header=False)
#         else:
#              # For features (DataFrames), save with header but without index
#             data.to_csv(f'data/processed_{name}.csv', index=False, header=True)

#     logger.info("Processed datasets saved to 'data/' directory.")

#     # Save the final feature list
#     with open('data/final_features.json', 'w') as f:
#         json.dump(final_features, f)
#     logger.info("Final feature list saved to 'data/final_features.json'")

#     # Save the label encoder if needed for inference
#     import joblib
#     joblib.dump(le_marital, 'data/label_encoder_marital.pkl')
#     logger.info("Label encoder saved to 'data/label_encoder_marital.pkl'")

#     return X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test, final_features


# # Example usage within the script (optional, can be called from notebook)
# # if __name__ == "__main__":
# #     filepath = "data/Participant_Selection_Final.csv" # Adjust path as needed
# #     X_tr, X_v, X_te, y1_tr, y1_v, ySuch_te, y2_tr, y2_v, y2_te, feats = run_feature_engineering(filepath)


# src/feature_engineering.py

import pandas as pd
import numpy as np
import logging
import json # Add import for saving feature list
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split # For splitting
from sklearn.ensemble import RandomForestClassifier # --- ADDED for feature selection ---
import warnings
warnings.filterwarnings('ignore') # Suppress potential warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads the dataset with correct headers and performs initial cleaning.
    """
    logger.info(f"Loading data from {filepath} (with headers)")
    df = pd.read_csv(filepath)

    logger.info(f"Initial data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Basic info
    logger.info(f"Data types:\n{df.dtypes}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")

    # --- Core Cleaning & Normalization ---
    # Convert Participant_Birthdate to datetime if possible
    df['Participant_Birthdate'] = pd.to_datetime(df['Participant_Birthdate'], errors='coerce')

    # Standardize booleans for categorical columns like Marrital_Status
    # Assuming 'Married' and 'Others(Widow/Unmarried)' are the values from the sample
    # Map to 0/1 if needed for consistency, or keep as categorical and encode later
    # For now, keep as categorical string, but ensure consistency
    df['Marrital_Status'] = df['Marrital_Status'].str.strip() # Remove leading/trailing spaces if any

    # Convert binary targets
    df['Participant_Selected_For_AID'] = df['Participant_Selected_For_AID'].map({'Yes': 1, 'No': 0}).astype('Int8')
    df['Aid_Type_Recomended'] = df['Aid_Type_Recomended'].str.strip() # Clean up potential whitespace

    return df


def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Applies the feature engineering plan.
    Returns the engineered DataFrame and the fitted LabelEncoder.
    """
    logger.info("Starting feature engineering...")
    df_fe = df.copy()

    # --- 3.1 Household structure & vulnerability ---
    # Assuming Family_Size is the total household size
    df_fe['hh_size'] = df_fe['Family_Size']

    # Dependency ratio (example: assuming 'has_under5' and 'has_50_plus' represent dependents)
    # This is a simplification; ideally, we'd have age bands
    df_fe['dependents_count'] = df_fe['has_under5'] + df_fe['has_50_plus']
    df_fe['working_age_adults'] = df_fe['has_18_50_Family_member'] # Assuming this is a count or flag for working age
    # Handle division by zero
    df_fe['dependency_ratio'] = df_fe['dependents_count'] / (df_fe['working_age_adults'] + 1e-8)

    # --- 3.2 Economic capacity & liquidity ---
    df_fe['income_total_monthly'] = df_fe['Income_Monthly_per_head'] * df_fe['hh_size']
    df_fe['expenditure_total_monthly'] = df_fe['Income_Monthly_per_head'] * df_fe['hh_size'] # Proxy
    df_fe['income_pc'] = df_fe['Income_Monthly_per_head']
    df_fe['expenditure_pc'] = df_fe['expenditure_total_monthly'] / df_fe['hh_size']
    df_fe['savings_to_expenditure'] = df_fe['Savings_Amt'] / (df_fe['expenditure_total_monthly'] + 1e-8)
    df_fe['debt_to_income'] = df_fe['Loans_Outstanding'] / (df_fe['Total_Income_Annualy'] + 1.0)
    df_fe['debt_to_income'] = df_fe['debt_to_income'].replace([float('inf'), float('-inf')], np.nan)
    df_fe['liquidity_gap'] = (df_fe['expenditure_total_monthly'] - df_fe['income_total_monthly']) / (df_fe['income_total_monthly'] + 1e-8)

    # --- 3.3 Assets & livelihood readiness ---
    df_fe['productive_asset_index'] = df_fe['Productive_Asset_Value']

    # --- 3.4 Health burden & care access ---
    df_fe['chronic_illness_count'] = df_fe['Chronic_Patient_Num']
    df_fe['disability_in_household'] = df_fe['Disabled_Yes']

    # --- 4. Aid-Type–Specific “Readiness/Need” Scores ---
    # Standardize inputs for scoring (using StandardScaler or min-max)
    scaler = StandardScaler()
    # Example features for each score (using available features, might need adjustment)
    cash_need_features = ['liquidity_gap', 'debt_to_income'] # Simplified for example
    cash_need_features = [f for f in cash_need_features if f in df_fe.columns] # Filter existing columns
    if cash_need_features:
        cash_need_scaled = scaler.fit_transform(df_fe[cash_need_features].fillna(0))
        df_fe['cash_need_score'] = cash_need_scaled.mean(axis=1) # Simple average, weights can be added later
    else:
        df_fe['cash_need_score'] = 0 # Default if no features available

    asset_readiness_features = ['productive_asset_index'] # Simplified for example
    asset_readiness_features = [f for f in asset_readiness_features if f in df_fe.columns]
    if asset_readiness_features:
        asset_readiness_scaled = scaler.fit_transform(df_fe[asset_readiness_features].fillna(0))
        df_fe['asset_readiness_score'] = asset_readiness_scaled.mean(axis=1)
    else:
        df_fe['asset_readiness_score'] = 0

    health_need_features = ['chronic_illness_count', 'disability_in_household'] # Simplified for example
    health_need_features = [f for f in health_need_features if f in df_fe.columns]
    if health_need_features:
        health_need_scaled = scaler.fit_transform(df_fe[health_need_features].fillna(0))
        df_fe['health_need_score'] = health_need_scaled.mean(axis=1)
    else:
        df_fe['health_need_score'] = 0

    training_suitability_features = ['Marrital_Status_Encoded'] # Placeholder, will be created below
    training_suitability_features = [f for f in training_suitability_features if f in df_fe.columns]
    if training_suitability_features:
        training_suitability_scaled = scaler.fit_transform(df_fe[training_suitability_features].fillna(0))
        df_fe['training_suitability_score'] = training_suitability_scaled.mean(axis=1)
    else:
        df_fe['training_suitability_score'] = 0

    # Create top need tag and softmax vector (requires all four scores to be non-zero, which they are now)
    score_cols = ['cash_need_score', 'asset_readiness_score', 'health_need_score', 'training_suitability_score']
    score_df = df_fe[score_cols]

    # Argmax for top need tag
    df_fe['top_need_tag'] = score_df.idxmax(axis=1)
    # Softmax for probability vector
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # Subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    softmax_vals = softmax(score_df.values)
    for i, col in enumerate(score_cols):
        df_fe[f'{col}_softmax'] = softmax_vals[:, i]

    # --- 5. Encodings & Interaction Strategy ---
    # Categorical encoding for Marrital_Status
    le_marital = LabelEncoder()
    df_fe['Marrital_Status_Encoded'] = le_marital.fit_transform(df_fe['Marrital_Status'].astype(str))

    # Interaction features (example)
    df_fe['dependency_ratio_x_income_pc'] = df_fe['dependency_ratio'] * df_fe['income_pc']
    df_fe['productive_asset_x_market_access'] = df_fe['productive_asset_index'] * 0 # Placeholder for market_access_proxy

    # Missingness indicators
    for col in ['income_pc', 'debt_to_income', 'savings_to_expenditure']:
        if col in df_fe.columns:
            df_fe[f'is_{col}_missing'] = df_fe[col].isna().astype(int)

    logger.info("Feature engineering completed.")
    return df_fe, le_marital # Return the encoder


def split_data(df_fe: pd.DataFrame, final_features: list, task1_target: str, task2_target: str, test_size: float = 0.2, val_size: float = 0.1):
    """
    Splits the data into train, validation, and test sets, stratifying by targets.
    Ensures X and y shapes are perfectly aligned before returning.
    """
    logger.info("Splitting data into train/val/test sets...")
    # Combine targets for stratification if both exist and are non-null in the same rows
    # Use rows where both targets are present for splitting if possible
    mask = df_fe[task1_target].notna() & df_fe[task2_target].notna()
    df_strat = df_fe[mask].copy()

    if df_strat.empty:
        logger.warning("No rows found where both targets are present for stratified splitting. Using simple split.")
        
        X_full = df_fe[final_features]
        
        y1_full = df_fe[task1_target]
        y2_full = df_fe[task2_target]

        # Debug: Check shapes before first split (simple split path)
        logger.info(f"Simple split path: Initial shapes - X: {X_full.shape}, y1: {y1_full.shape}, y2: {y2_full.shape}")
        if X_full.shape[0] != y1_full.shape[0] or X_full.shape[0] != y2_full.shape[0]:
            logger.error("FATAL ERROR: Initial shapes do not match before simple split.")
            logger.error(f"  X: {X_full.shape}, y1: {y1_full.shape}, y2: {y2_full.shape}")
            raise ValueError("Initial X, y1, y2 shapes do not match for simple split.")

        # Simple split for Task 1 target (stratify on y1)
        # train_test_split should maintain alignment between X_out and y_out
        X_temp, X_test, y1_temp, y1_test, y2_temp, y2_test = train_test_split(
            X_full, y1_full, y2_full, test_size=test_size, random_state=42, stratify=y1_full
        )
        logger.info(f"Shapes after first split (temp/test): X_temp: {X_temp.shape}, y1_temp: {y1_temp.shape}, y2_temp: {y2_temp.shape}")
        logger.info(f"Shapes after first split (temp/test): X_test: {X_test.shape}, y1_test: {y1_test.shape}, y2_test: {y2_test.shape}")

        # Verify alignment after first split
        if not (X_temp.shape[0] == y1_temp.shape[0] == y2_temp.shape[0]):
            logger.error(f"Shape mismatch after first split (temp): X {X_temp.shape[0]}, y1 {y1_temp.shape[0]}, y2 {y2_temp.shape[0]}")
            raise ValueError("Shape mismatch after first split (temp).")
        if not (X_test.shape[0] == y1_test.shape[0] == y2_test.shape[0]):
            logger.error(f"Shape mismatch after first split (test): X {X_test.shape[0]}, y1 {y1_test.shape[0]}, y2 {y2_test.shape[0]}")
            raise ValueError("Shape mismatch after first split (test).")

        # Split temp into train and val (stratify on y1_temp)
        X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
            X_temp, y1_temp, y2_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y1_temp
        )
        logger.info(f"Shapes after second split (train/val): X_train: {X_train.shape}, y1_train: {y1_train.shape}, y2_train: {y2_train.shape}")
        logger.info(f"Shapes after second split (train/val): X_val: {X_val.shape}, y1_val: {y1_val.shape}, y2_val: {y2_val.shape}")

        # Verify alignment after second split
        if not (X_train.shape[0] == y1_train.shape[0] == y2_train.shape[0]):
            logger.error(f"Shape mismatch after second split (train): X {X_train.shape[0]}, y1 {y1_train.shape[0]}, y2 {y2_train.shape[0]}")
            raise ValueError("Shape mismatch after second split (train).")
        if not (X_val.shape[0] == y1_val.shape[0] == y2_val.shape[0]):
            logger.error(f"Shape mismatch after second split (val): X {X_val.shape[0]}, y1 {y1_val.shape[0]}, y2 {y2_val.shape[0]}")
            raise ValueError("Shape mismatch after second split (val).")

    else:
        logger.info(f"Found {df_strat.shape[0]} rows with both targets. Using stratified split on these rows.")
        
        X_strat = df_strat[final_features]

        y_strat_df = df_strat[[task1_target, task2_target]] # Keep as DataFrame for combined strat
        # Create combined strat column string
        y_strat_combined = y_strat_df.apply(lambda x: f"{x[task1_target]}_{x[task2_target]}", axis=1)

        logger.info(f"Shapes for stratified split: X_strat: {X_strat.shape}, y_strat_combined: {y_strat_combined.shape}")

        X_temp, X_test, y_temp_df, y_test_df = train_test_split(
            X_strat, y_strat_df, test_size=test_size, random_state=42, stratify=y_strat_combined
        )
        logger.info(f"Shapes after first strat split (temp/test): X_temp: {X_temp.shape}, y_temp_df: {y_temp_df.shape}, y_test_df: {y_test_df.shape}")

        # Extract y1, y2 from the resulting DataFrames
        y1_temp = y_temp_df[task1_target]
        y2_temp = y_temp_df[task2_target]
        y1_test = y_test_df[task1_target]
        y2_test = y_test_df[task2_target]
        logger.info(f"Shapes after extracting y from temp/test: y1_temp: {y1_temp.shape}, y2_temp: {y2_temp.shape}, y1_test: {y1_test.shape}, y2_test: {y2_test.shape}")

        # Verify alignment after first strat split
        if not (X_temp.shape[0] == y1_temp.shape[0] == y2_temp.shape[0]):
            logger.error(f"Shape mismatch after first strat split (temp): X {X_temp.shape[0]}, y1 {y1_temp.shape[0]}, y2 {y2_temp.shape[0]}")
            raise ValueError("Shape mismatch after first strat split (temp).")
        if not (X_test.shape[0] == y1_test.shape[0] == y2_test.shape[0]):
            logger.error(f"Shape mismatch after first strat split (test): X {X_test.shape[0]}, y1 {y1_test.shape[0]}, y2 {y2_test.shape[0]}")
            raise ValueError("Shape mismatch after first strat split (test).")

        # Recreate combined strat column for temp split
        y_strat_temp_combined = y_temp_df.apply(lambda x: f"{x[task1_target]}_{x[task2_target]}", axis=1)
        X_train, X_val, y_train_df, y_val_df = train_test_split(
            X_temp, y_temp_df, test_size=val_size/(1-test_size), random_state=42, stratify=y_strat_temp_combined
        )
        logger.info(f"Shapes after second strat split (train/val): X_train: {X_train.shape}, y_train_df: {y_train_df.shape}, y_val_df: {y_val_df.shape}")

        # Extract y1, y2 from the resulting DataFrames for train/val
        y1_train = y_train_df[task1_target]
        y2_train = y_train_df[task2_target]
        y1_val = y_val_df[task1_target]
        y2_val = y_val_df[task2_target]
        logger.info(f"Shapes after extracting y from train/val: y1_train: {y1_train.shape}, y2_train: {y2_train.shape}, y1_val: {y1_val.shape}, y2_val: {y2_val.shape}")

        # Verify alignment after second strat split
        if not (X_train.shape[0] == y1_train.shape[0] == y2_train.shape[0]):
            logger.error(f"Shape mismatch after second strat split (train): X {X_train.shape[0]}, y1 {y1_train.shape[0]}, y2 {y2_train.shape[0]}")
            raise ValueError("Shape mismatch after second strat split (train).")
        if not (X_val.shape[0] == y1_val.shape[0] == y2_val.shape[0]):
            logger.error(f"Shape mismatch after second strat split (val): X {X_val.shape[0]}, y1 {y1_val.shape[0]}, y2 {y2_val.shape[0]}")
            raise ValueError("Shape mismatch after second strat split (val).")

    # Debug: Check shapes before returning
    logger.info(f"Final split shapes - Train: X={X_train.shape}, y1={y1_train.shape}, y2={y2_train.shape}")
    logger.info(f"Final split shapes - Val: X={X_val.shape}, y1={y1_val.shape}, y2={y2_val.shape}")
    logger.info(f"Final split shapes - Test: X={X_test.shape}, y1={y1_test.shape}, y2={y2_test.shape}")

    return (X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test)


def run_feature_engineering(filepath: str) -> tuple:
    """
    Main function to run the feature engineering pipeline.
    Returns split datasets and feature info.
    """
    logger.info("Starting Feature Engineering Pipeline...")

    df = load_and_clean_data(filepath)
    df_fe, le_marital = engineer_features(df) # Get the encoder

    task1_target = 'Participant_Selected_For_AID'
    task2_target = 'Aid_Type_Recomended'

    # Identify final features (exclude targets and potentially ID columns if present)
    exclude_cols = [task1_target, task2_target, 'Sl', 'Participant_ID', 'Participant_Birthdate', 
                    'Marrital_Status', 'top_need_tag'] # Add other non-numeric/original string cols
    all_engineered_features = [col for col in df_fe.columns if col not in exclude_cols]
    
    # Ensure all final features are numeric, handling potential NaNs
    df_fe[all_engineered_features] = df_fe[all_engineered_features].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    logger.info(f"Full feature list ({len(all_engineered_features)}): {all_engineered_features[:10]}...") # Print first 10 as example

    # Pass all_engineered_features to split_data
    X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test = \
        split_data(df_fe, all_engineered_features, task1_target, task2_target)

    # --- START FEATURE SELECTION (Top 10) ---
    logger.info("Starting feature selection based on Task 1...")
    
    # Handle NaNs in target if any (though stratification should have handled it)
    # Re-combine X_train and y1_train to safely drop NaNs in target
    temp_train_df = X_train.copy()
    temp_train_df['target'] = y1_train
    temp_train_df = temp_train_df.dropna(subset=['target'])
    
    X_train_fs = temp_train_df.drop(columns=['target'])
    y1_train_fs = temp_train_df['target']
    
    # Ensure target is integer
    y1_train_fs = y1_train_fs.astype(int)

    # Train a RandomForest to find feature importances
    fs_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    fs_model.fit(X_train_fs, y1_train_fs)
    
    # Get importances
    importances = fs_model.feature_importances_
    importance_series = pd.Series(importances, index=all_engineered_features)
    
    # Select top 10 features
    top_10_features = importance_series.nlargest(10).index.tolist()
    
    logger.info(f"Top 10 features selected: {top_10_features}")

    # Filter all datasets to only these 10 features
    X_train = X_train[top_10_features]
    X_val = X_val[top_10_features]
    X_test = X_test[top_10_features]
    
    # Assign the final feature list to be saved
    final_features_to_save = top_10_features
    # --- END FEATURE SELECTION ---

    # Debug: Check shapes before saving
    logger.info(f"Shapes before saving (Top 10) - Train: X={X_train.shape}, y1={y1_train.shape}, y2={y2_train.shape}")
    logger.info(f"Shapes before saving (Top 10) - Val: X={X_val.shape}, y1={y1_val.shape}, y2={y2_val.shape}")
    logger.info(f"Shapes before saving (Top 10) - Test: X={X_test.shape}, y1={y1_test.shape}, y2={y2_test.shape}")

    # Verify shapes align before saving
    assert X_train.shape[0] == y1_train.shape[0] == y2_train.shape[0], f"Shape mismatch in Train split: X {X_train.shape[0]}, y1 {y1_train.shape[0]}, y2 {y2_train.shape[0]}"
    assert X_val.shape[0] == y1_val.shape[0] == y2_val.shape[0], f"Shape mismatch in Val split: X {X_val.shape[0]}, y1 {y1_val.shape[0]}, y2 {y2_val.shape[0]}"
    assert X_test.shape[0] == y1_test.shape[0] == y2_test.shape[0], f"Shape mismatch in Test split: X {X_test.shape[0]}, y1 {y1_test.shape[0]}, y2 {y2_test.shape[0]}"

    # Save the processed datasets
    processed_data_dict = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y1_train': y1_train,
        'y1_val': y1_val,
        'y1_test': y1_test,
        'y2_train': y2_train,
        'y2_val': y2_val,
        'y2_test': y2_test
    }
    
    # Add index=False to all .to_csv() calls
    for name, data in processed_data_dict.items():
        if name.startswith('y'):
             # For targets (Series), save without header and without index
            data.to_csv(f'data/processed_{name}.csv', index=False, header=False)
        else:
             # For features (DataFrames), save with header but without index
            data.to_csv(f'data/processed_{name}.csv', index=False, header=True)

    logger.info("Processed datasets saved to 'data/' directory.")

    # Save the final (top 10) feature list
    with open('data/final_features.json', 'w') as f:
        json.dump(final_features_to_save, f)
    logger.info(f"Top 10 feature list saved to 'data/final_features.json'")

    # Save the label encoder if needed for inference
    import joblib
    joblib.dump(le_marital, 'data/label_encoder_marital.pkl')
    logger.info("Label encoder saved to 'data/label_encoder_marital.pkl'")

    return X_train, X_val, X_test, y1_train, y1_val, y1_test, y2_train, y2_val, y2_test, final_features_to_save


# Example usage within the script (optional, can be called from notebook)
# if __name__ == "__main__":
#     filepath = "data/Participant_Selection_Final.csv" # Adjust path as needed
#     X_tr, X_v, X_te, y1_tr, y1_v, y1_te, y2_tr, y2_v, y2_te, feats = run_feature_engineering(filepath)