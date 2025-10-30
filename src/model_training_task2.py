# src/model_training_task2.py

import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler # Might be needed depending on the model
import warnings
warnings.filterwarnings('ignore') # Suppress potential warnings from sklearn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

def load_processed_data(task_name: str = "train") -> tuple:
    """
    Loads the processed dataset splits for Task 2.
    """
    logger.info(f"Loading processed {task_name} data for Task 2...")
    try:
        X = pd.read_csv(f'data/processed_X_{task_name}.csv')
        y_df = pd.read_csv(f'data/processed_y2_{task_name}.csv', header=None) # Load as DataFrame
        y = y_df.iloc[:, 0] # Select the first column as a Series
        # Load the eligibility probability from Task 1 (assuming it was saved as a CSV with one column)
        # The name of the file depends on how task1 saved it. Let's assume it saved just the probabilities.
        proba_df = pd.read_csv(f'data/task1_{task_name}_proba_calibrated_run_1.csv', header=None) # Load as DataFrame
        proba = proba_df.iloc[:, 0] # Select the first column as a Series
        # Add the probability as a feature
        X_with_proba = X.copy()
        X_with_proba['eligibility_proba'] = proba.values # Ensure alignment
        logger.info(f"Task 2 {task_name} data loaded. Shape: {X_with_proba.shape}, Target shape: {y.shape}")
        
        # --- Handle potential NaNs in target (e.g., from rows without Aid_Type_Recomended) ---
        # This is crucial for class weight calculation
        valid_mask = y.notna()
        X_with_proba = X_with_proba[valid_mask]
        y = y[valid_mask]
        logger.info(f"Data filtered for valid targets. New shape: {X_with_proba.shape}, Target shape: {y.shape}")
        
        return X_with_proba, y
    except FileNotFoundError as e:
        logger.error(f"Processed {task_name} data file or Task 1 probabilities not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading Task 2 {task_name} data: {e}")
        raise


def evaluate_model(model, X_test, y_test, model_name: str, run_id: str = "default"):
    """
    Evaluates a trained model using various metrics for multi-class classification.
    """
    logger.info(f"Evaluating model: {model_name}")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    # Use 'macro' averaging for precision, recall, and F1 to get unweighted mean across classes
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    metrics = {
        'Model': model_name,
        'Run_ID': run_id,
        'Accuracy': accuracy,
        'Precision (Macro)': precision,
        'Recall (Macro)': recall,
        'F1-Score (Macro)': f1
    }

    logger.info(f"Results for {model_name}:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision (Macro): {precision:.4f}")
    logger.info(f"  Recall (Macro): {recall:.4f}")
    logger.info(f"  F1-Score (Macro): {f1:.4f}")

    # Print detailed classification report
    logger.info(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}")
    logger.info(f"Confusion Matrix for {model_name}:\n{confusion_matrix(y_test, y_pred)}")

    return metrics, y_pred


def train_and_evaluate_ensemble_models(X_train, X_val, y_train, y_val, run_id: str = "default"):
    """
    Trains ensemble models for Task 2.
    """
    logger.info("Starting training and evaluation of ensemble models for Task 2...")
    # Apply class weights if imbalanced
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    logger.info(f"Computed class weights for Task 2: {class_weight_dict}")

    models = {
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42), # Does not accept class_weight
        'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict),
    }

    # Train base models and collect results
    trained_models = {}
    results = []
    predictions = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        metrics, y_pred = evaluate_model(model, X_val, y_val, name, run_id)
        results.append(metrics)
        predictions[name] = y_pred

    # Train Voting Classifier
    logger.info("Training VotingClassifier...")

    # --- FIX ---
    # Define NEW, unfitted estimators for the VotingClassifier.
    # Set class_weight='balanced' so they compute weights on the
    # numeric labels [0,1,2] that VotingClassifier passes them internally.
    # Do NOT pass trained_models[...] here.
    
    clf1_vc = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf2_vc = GradientBoostingClassifier(random_state=42)
    clf3_vc = ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    voting_clf = VotingClassifier(estimators=[
        ('rf', clf1_vc),
        ('gb', clf2_vc),
        ('et', clf3_vc)
    ], voting='hard') # Use 'hard' voting for final class prediction
    # --- END FIX ---
    
    voting_clf.fit(X_train, y_train)
    trained_models['VotingClassifier'] = voting_clf

    metrics, y_pred = evaluate_model(voting_clf, X_val, y_val, 'VotingClassifier', run_id)
    results.append(metrics)
    predictions['VotingClassifier'] = y_pred

    return trained_models, results, predictions


def save_models_and_results(trained_models: dict, results: list, run_id: str = "default"):
    """
    Saves the trained models and evaluation results for Task 2.
    """
    logger.info("Saving models and results for Task 2...")

    # Save individual models
    for name, model in trained_models.items():
        model_filename = f"models/task2_{name}_{run_id}.pkl"
        joblib.dump(model, model_filename)
        logger.info(f"Model {name} saved to {model_filename}")

    # Save evaluation results
    results_df = pd.DataFrame(results)
    results_filename = f"results/task2_evaluation_{run_id}.csv"
    results_df.to_csv(results_filename, index=False)
    logger.info(f"Evaluation results saved to {results_filename}")

    # Save a summary report
    summary_filename = f"results/task2_summary_{run_id}.txt"
    with open(summary_filename, 'w') as f:
        f.write("Task 2 Model Evaluation Summary\n")
        f.write("="*40 + "\n")
        for result in results:
            f.write(f"Model: {result['Model']}\n")
            f.write(f"  Accuracy: {result['Accuracy']:.4f}\n")
            f.write(f"  Precision (Macro): {result['Precision (Macro)']:.4f}\n")
            f.write(f"  Recall (Macro): {result['Recall (Macro)']:.4f}\n")
            f.write(f"  F1-Score (Macro): {result['F1-Score (Macro)']:.4f}\n")
            f.write("-" * 30 + "\n")
    logger.info(f"Summary report saved to {summary_filename}")


def run_model_training_task2(
    run_id: str = "run_1"
):
    """
    Main function to run the model training pipeline for Task 2.
    Uses features from feature engineering and eligibility probability from Task 1.
    """
    logger.info(f"Starting Task 2 Model Training Pipeline (Run ID: {run_id})...")

    X_train, y2_train = load_processed_data("train")
    X_val, y2_val = load_processed_data("val")

    # Train and evaluate models
    trained_models, results, predictions = train_and_evaluate_ensemble_models(X_train, X_val, y2_train, y2_val, run_id)

    # Save models and results
    save_models_and_results(trained_models, results, run_id)

    logger.info("Task 2 Model Training Pipeline completed.")

    # Return the best performing model based on a chosen metric (e.g., F1-Score)
    # Find the model with the highest F1-Score (Macro Average)
    best_result = max(results, key=lambda x: x['F1-Score (Macro)'])
    best_model_name = best_result['Model']
    best_model = trained_models[best_model_name]
    logger.info(f"Best performing model based on F1-Score (Macro): {best_model_name}")

    return trained_models, results, predictions, X_val, y2_val, best_model, best_model_name


# Example usage within the script (optional, can be called from notebook)
# if __name__ == "__main__":
#     models, eval_results, preds, X_v, y2_v, best_mdl, best_name = run_model_training_task2(run_id="with_eligibility_proba_run_1")