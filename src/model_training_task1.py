# src/model_training_task1.py

import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression # Add Logistic Regression
from sklearn.calibration import CalibratedClassifierCV # For calibrated probabilities
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
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
    Loads the processed dataset splits for Task 1.
    """
    logger.info(f"Loading processed {task_name} data for Task 1...")
    try:
        X = pd.read_csv(f'data/processed_X_{task_name}.csv')
        # Load target column as a Series by selecting the first column after index (if index was saved)
        # Or assume the file has only one column and select it directly.
        # Let's assume the file has only the target column and no index column saved separately.
        y_df = pd.read_csv(f'data/processed_y1_{task_name}.csv', header=None) # Load as DataFrame
        y = y_df.iloc[:, 0] # Select the first column as a Series
        logger.info(f"Task 1 {task_name} data loaded. Shape: {X.shape}, Target shape: {y.shape}")
        # Debug: Check if shapes match
        if X.shape[0] != y.shape[0]:
             logger.error(f"SHAPE MISMATCH in {task_name}: X has {X.shape[0]} rows, y has {y.shape[0]} rows.")
             raise ValueError(f"Shape mismatch in {task_name}: X ({X.shape}) vs y ({y.shape})")
        return X, y
    except FileNotFoundError as e:
        logger.error(f"Processed {task_name} data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading Task 1 {task_name}  {e}")
        raise


def evaluate_model(model, X_test, y_test, model_name: str, run_id: str = "default"):
    """
    Evaluates a trained model using various metrics.
    """
    logger.info(f"Evaluating model: {model_name}")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None # Get probability for positive class

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else float('nan')

    metrics = {
        'Model': model_name,
        'Run_ID': run_id,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    }

    logger.info(f"Results for {model_name}:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}" if not pd.isna(roc_auc) else "  ROC-AUC: N/A (Probability scores not available)")

    # Print detailed classification report
    logger.info(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}")
    logger.info(f"Confusion Matrix for {model_name}:\n{confusion_matrix(y_test, y_pred)}")

    return metrics, y_pred, y_pred_proba


def train_and_evaluate_ensemble_models(X_train, X_val, y_train, y_val, run_id: str = "default"):
    """
    Trains ensemble models with calibration for Task 1.
    """
    logger.info("Starting training and evaluation of ensemble models for Task 1 with calibration...")

    models = {
        'RandomForestClassifier': CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, random_state=42), cv=3),
        'GradientBoostingClassifier': CalibratedClassifierCV(GradientBoostingClassifier(random_state=42), cv=3),
        'ExtraTreesClassifier': CalibratedClassifierCV(ExtraTreesClassifier(n_estimators=100, random_state=42), cv=3),
        'LogisticRegression_L1': CalibratedClassifierCV(LogisticRegression(penalty='l1', solver='liblinear', random_state=42), cv=3),
    }

    # Train base models and collect results
    trained_models = {}
    results = []
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        logger.info(f"Training {name} (with calibration)...")
        # Debug: Check shapes before fit
        logger.info(f"  Fitting {name}: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Evaluate on validation set
        metrics, y_pred, y_pred_proba = evaluate_model(model, X_val, y_val, name, run_id)
        results.append(metrics)
        predictions[name] = y_pred
        if y_pred_proba is not None:
            probabilities[name] = y_pred_proba

    # Train Voting Classifier
    logger.info("Training VotingClassifier...")
    # Note: VotingClassifier with 'soft' voting needs calibrated base estimators or predict_proba support
    # Since CalibratedClassifierCV adds predict_proba, this should work.
    voting_clf = VotingClassifier(estimators=[
        ('rf', trained_models['RandomForestClassifier']),
        ('gb', trained_models['GradientBoostingClassifier']),
        ('et', trained_models['ExtraTreesClassifier']),
        ('lr_l1', trained_models['LogisticRegression_L1'])
    ], voting='soft') # Use 'soft' voting with calibrated probabilities
    voting_clf.fit(X_train, y_train)
    trained_models['VotingClassifier'] = voting_clf

    # Evaluate Voting Classifier on validation set
    metrics, y_pred, y_pred_proba = evaluate_model(voting_clf, X_val, y_val, 'VotingClassifier', run_id)
    results.append(metrics)
    predictions['VotingClassifier'] = y_pred
    if y_pred_proba is not None:
        probabilities['VotingClassifier'] = y_pred_proba

    return trained_models, results, predictions, probabilities


def save_models_and_results(trained_models: dict, results: list, run_id: str = "default"):
    """
    Saves the trained models and evaluation results for Task 1.
    """
    logger.info("Saving models and results for Task 1...")

    # Save individual models
    for name, model in trained_models.items():
        model_filename = f"models/task1_{name}_{run_id}.pkl"
        joblib.dump(model, model_filename)
        logger.info(f"Model {name} saved to {model_filename}")

    # Save evaluation results
    results_df = pd.DataFrame(results)
    results_filename = f"results/task1_evaluation_{run_id}.csv"
    results_df.to_csv(results_filename, index=False)
    logger.info(f"Evaluation results saved to {results_filename}")

    # Save a summary report
    summary_filename = f"results/task1_summary_{run_id}.txt"
    with open(summary_filename, 'w') as f:
        f.write("Task 1 Model Evaluation Summary\n")
        f.write("="*40 + "\n")
        for result in results:
            f.write(f"Model: {result['Model']}\n")
            f.write(f"  Accuracy: {result['Accuracy']:.4f}\n")
            f.write(f"  Precision: {result['Precision']:.4f}\n")
            f.write(f"  Recall: {result['Recall']:.4f}\n")
            f.write(f"  F1-Score: {result['F1-Score']:.4f}\n")
            f.write(f"  ROC-AUC: {result['ROC-AUC']:.4f}\n" if not pd.isna(result['ROC-AUC']) else f"  ROC-AUC: N/A\n")
            f.write("-" * 30 + "\n")
    logger.info(f"Summary report saved to {summary_filename}")


def run_model_training_task1(
    run_id: str = "run_1"
):
    """
    Main function to run the model training pipeline for Task 1.
    Includes calibration for Task 2 use.
    """
    logger.info(f"Starting Task 1 Model Training Pipeline (Run ID: {run_id})...")

    X_train, y1_train = load_processed_data("train")
    X_val, y1_val = load_processed_data("val")

    # Train and evaluate models
    trained_models, results, predictions, probabilities = train_and_evaluate_ensemble_models(X_train, X_val, y1_train, y1_val, run_id)

    # Save models and results
    save_models_and_results(trained_models, results, run_id)

    logger.info("Task 1 Model Training Pipeline completed.")

    # Return the best performing model based on a chosen metric (e.g., ROC-AUC)
    # Find the model with the highest ROC-AUC
    best_result = max(results, key=lambda x: x['ROC-AUC'])
    best_model_name = best_result['Model']
    best_model = trained_models[best_model_name]
    logger.info(f"Best performing model based on ROC-AUC: {best_model_name}")

    # Generate calibrated probabilities for the training set to be used as a feature in Task 2
    logger.info("Generating calibrated probabilities for Task 2 features...")
    best_model_on_train = trained_models[best_model_name] # Get the best model object
    train_proba = best_model_on_train.predict_proba(X_train)[:, 1]
    val_proba = best_model_on_train.predict_proba(X_val)[:, 1]
    
    # --- FIX ---
    # Added header=False to both .to_csv() calls
    pd.Series(train_proba, name='eligibility_proba').to_csv(f'data/task1_train_proba_{run_id}.csv', index=False, header=False)
    pd.Series(val_proba, name='eligibility_proba').to_csv(f'data/task1_val_proba_{run_id}.csv', index=False, header=False)
    # --- END FIX ---
    
    logger.info(f"Task 1 calibrated probabilities (train, val) saved.")

    return trained_models, results, predictions, probabilities, X_val, y1_val, best_model, best_model_name


# Example usage within the script (optional, can be called from notebook)
# if __name__ == "__main__":
#     models, eval_results, preds, probs, X_v, y1_v, best_mdl, best_name = run_model_training_task1(run_id="calibrated_run_1")