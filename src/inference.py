# src/inference.py

import pandas as pd
import numpy as np
import joblib
import json
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- NEW: Define thresholds for heuristic rules ---
# WARNING: These are example thresholds. You MUST tune these based on your
# training data's distribution (e.g., using the 75th percentile).
HEALTH_NEED_THRESHOLD = 0.5 # Example: If health_need_score_softmax > 0.5
CASH_NEED_THRESHOLD = 0.0   # Example: If cash_need_score is positive
logger.warning(f"Using example thresholds for logic: HEALTH_NEED_THRESHOLD={HEALTH_NEED_THRESHOLD}, CASH_NEED_THRESHOLD={CASH_NEED_THRESHOLD}")


def load_model_and_resources(
    model_task1_path: str = 'models/task1_VotingClassifier_run_1.pkl',
    model_task2_path: str = 'models/task2_VotingClassifier_run_1.pkl',
    features_path: str = 'data/final_features.json',
    label_encoder_path: str = 'data/label_encoder_marital.pkl'
) -> tuple:
    """
    Loads the trained models, features, and label encoder.
    """
    logger.info("Loading models and resources for inference...")

    # Load models
    try:
        model_task1 = joblib.load(model_task1_path)
        logger.info(f"Task 1 model loaded from {model_task1_path}")
    except FileNotFoundError:
        logger.error(f"Task 1 model file not found: {model_task1_path}")
        model_task1 = None
    except Exception as e:
        logger.error(f"Error loading Task 1 model: {e}")
        model_task1 = None

    try:
        model_task2 = joblib.load(model_task2_path)
        logger.info(f"Task 2 model loaded from {model_task2_path}")
    except FileNotFoundError:
        logger.error(f"Task 2 model file not found: {model_task2_path}")
        model_task2 = None
    except Exception as e:
        logger.error(f"Error loading Task 2 model: {e}")
        model_task2 = None

    # Load features
    try:
        with open(features_path, 'r') as f:
            features = json.load(f)
        logger.info(f"Features loaded from {features_path}")
    except FileNotFoundError:
        logger.error(f"Features file not found: {features_path}")
        features = None
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        features = None

    # Load label encoder
    try:
        le_marital = joblib.load(label_encoder_path)
        logger.info(f"Label encoder loaded from {label_encoder_path}")
    except FileNotFoundError:
        logger.error(f"Label encoder file not found: {label_encoder_path}")
        le_marital = None # This is not strictly needed if we assume processed data
    except Exception as e:
        logger.error(f"Error loading label encoder: {e}")
        le_marital = None

    return model_task1, model_task2, features, le_marital


def preprocess_new_data(df: pd.DataFrame, features: list):
    """
    Prepares already-processed data for inference.
    Assumes 'df' is processed (like X_test).
    """
    logger.info("Preprocessing new data...")
    df_processed = df.copy() # Work on a copy

    # Fill missing values for the specified feature columns
    # (In case new data has NaNs that training data didn't)
    for col in features:
        if col in df_processed.columns:
            if df_processed[col].dtype in ['int64', 'float64']:
                # Fill NaNs with 0 (or median/mean from training if saved)
                df_processed[col].fillna(0, inplace=True)
        else:
            # If a feature is missing entirely, add it and fill with 0
            logger.warning(f"Feature '{col}' not found in new data. Adding and filling with 0.")
            df_processed[col] = 0

    # Ensure the dataframe has the correct columns in the right order for the model
    try:
        X_new = df_processed[features].copy()
        logger.info(f"Feature matrix shape after selection: {X_new.shape}")
    except KeyError as e:
        logger.error(f"KeyError during feature selection: {e}")
        logger.error(f"Available columns in df_processed: {list(df_processed.columns)}")
        logger.error(f"Required features: {features}")
        return None # Indicate failure

    logger.info(f"Preprocessing successful. Shape of feature matrix: {X_new.shape}")
    return X_new


def run_inference(
    new_data: pd.DataFrame,
    model_task1_path: str = 'models/task1_VotingClassifier_calibrated_run_1.pkl', # Use calibrated model
    model_task2_path: str = 'models/task2_VotingClassifier_with_eligibility_proba_run_1.pkl',
    features_path: str = 'data/final_features.json',
    label_encoder_path: str = 'data/label_encoder_marital.pkl'
):
    """
    Runs inference for both Task 1 and Task 2 on new data.
    """
    logger.info("Starting inference pipeline...")

    model_task1, model_task2, features, le_marital = load_model_and_resources(
        model_task1_path, model_task2_path, features_path, label_encoder_path
    )

    if not all([model_task1, model_task2, features]): # le_marital not strictly needed now
        logger.error("Failed to load necessary models, resources, or feature list. Aborting inference.")
        return new_data.copy() # Return original data if preprocessing fails, indicating no predictions

    X_new = preprocess_new_data(new_data, features)
    
    if X_new is None:
        logger.error("Preprocessing failed. Aborting inference.")
        return new_data.copy()

    # --- Task 1 Inference ---
    logger.info("Running Task 1 inference (Eligibility Probability)...")
    task1_probabilities = None
    task1_predictions = None
    try:
        # Get probability for positive class (Eligible = 1) - calibrated
        task1_probabilities = model_task1.predict_proba(X_new)[:, 1]
        task1_predictions = model_task1.predict(X_new) # Also get the binary prediction if needed
        logger.info(f"Task 1 inference completed successfully.")
    except Exception as e:
        logger.error(f"Error during Task 1 inference: {e}")
        logger.error(f"X_new shape: {X_new.shape}")
        logger.error(f"X_new dtypes:\n{X_new.dtypes}")
        # task1_probabilities and task1_predictions remain None

    # --- Task 2 Inference ---
    logger.info("Running Task 2 inference (Aid Type Recommendation)...")
    task2_predictions = None
    task2_probabilities_all_classes = None
    task2_classes = None
    try:
        # Add Task 1 probability as a feature for Task 2 model
        if task1_probabilities is not None:
            X_new_with_proba = X_new.copy()
            X_new_with_proba['eligibility_proba'] = task1_probabilities
            
            task2_predictions = model_task2.predict(X_new_with_proba)
            # Get prediction probabilities for all classes (if available)
            if hasattr(model_task2, "predict_proba"):
                 task2_probabilities_all_classes = model_task2.predict_proba(X_new_with_proba)
                 # Get the list of classes the model was trained on
                 task2_classes = model_task2.classes_
            logger.info(f"Task 2 inference completed successfully.")
        else:
            logger.error("Task 1 probabilities not available, cannot run Task 2 inference.")
    except Exception as e:
        logger.error(f"Error during Task 2 inference: {e}")
        logger.error(f"X_new_with_proba shape: {X_new_with_proba.shape if 'X_new_with_proba' in locals() else 'N/A'}")
        if 'eligibility_proba' in X_new.columns:
             logger.error("Potential problem: 'eligibility_proba' was already in X_new. Check 'final_features.json'.")
        # task2_predictions, task2_probabilities_all_classes, task2_classes remain None

    # --- Prepare Results ---
    results_df = new_data.copy()

    # Add Task 1 results if available
    if task1_probabilities is not None:
        results_df['Task1_Eligibility_Probability'] = task1_probabilities
        results_df['Task1_Eligible_Prediction'] = task1_predictions
        logger.info("Task 1 results added to results_df.")
    else:
        logger.warning("Task 1 results were not generated, columns not added.")

    # Add Task 2 results if available
    if task2_predictions is not None and task2_probabilities_all_classes is not None and task2_classes is not None:
        results_df['Task2_Recommended_Aid_Type'] = task2_predictions
        # Add probability for each class as a separate column
        for i, class_name in enumerate(task2_classes):
            col_name = f'Task2_Probability_{class_name.replace(" ", "_")}'
            results_df[col_name] = task2_probabilities_all_classes[:, i]
        logger.info("Task 2 results added to results_df.")
    else:
        logger.warning("Task 2 results were not generated, columns not added.")


    # --- START NEW LOGIC BLOCK: Apply Business Rules and Mapping ---
    logger.info("Applying final business logic to map recommendations...")

    final_recommendations = []
    
    # Iterate over each row in the inference results
    for i in range(len(X_new)):
        # Get the feature scores for the current household
        # We use X_new here as it's the processed feature set
        health_score = X_new.iloc[i]['health_need_score_softmax']
        cash_score = X_new.iloc[i]['cash_need_score']
        
        # Get the ML model's recommendation from the results_df
        ml_recommendation = results_df.iloc[i]['Task2_Recommended_Aid_Type']
        
        # 1. Check for critical Health Support (Highest priority)
        if health_score > HEALTH_NEED_THRESHOLD:
            final_recommendations.append('Health Support')
            
        # 2. Check for critical Cash Grant (Second priority)
        elif cash_score > CASH_NEED_THRESHOLD:
            final_recommendations.append('Cash Grant')
            
        # 3. Map the ML Model's output to business goals
        elif ml_recommendation == 'Enterprise Development':
            final_recommendations.append('Livelihood Asset')
            
        elif ml_recommendation == 'Skill Development':
            final_recommendations.append('Training')
            
        elif ml_recommendation == 'Both':
            final_recommendations.append('Livelihood Asset AND Training')
            
        else:
            final_recommendations.append('N/A') # Default case

    # Add the final, mapped recommendations to the results
    results_df['Final_Recommended_Aid'] = final_recommendations
    logger.info("Final recommendation mapping completed.")
    # --- END NEW LOGIC BLOCK ---

    logger.info("Inference pipeline completed.")
    return results_df