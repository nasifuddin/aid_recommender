import pandas as pd
import numpy as np
import joblib
import json
import logging
import shap
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# --- 1. Setup & Configuration ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 2. Re-implement Feature Engineering Logic ---
# This is the same logic from your notebook's inference cell
def engineer_features_for_inference(df: pd.DataFrame, le_marital: LabelEncoder) -> pd.DataFrame:
    """
    Applies the same feature engineering plan used during training to new raw data.
    """
    logging.info("Applying feature engineering to new data...")
    df_fe = df.copy()
    
    # Fill NaNs in base columns with 0 before calculations
    numeric_cols = ['Family_Size', 'has_under5', 'has_50_plus', 'has_18_50_Family_member', 
                    'Income_Monthly_per_head', 'Savings_Amt', 'Total_Income_Annualy', 
                    'Loans_Outstanding', 'Productive_Asset_Value', 'Chronic_Patient_Num', 
                    'Disabled_Yes', 'Loans_Running_Yes', 'Loans_Num', 'has_Savings']
    for col in numeric_cols:
        if col in df_fe.columns:
            df_fe[col] = pd.to_numeric(df_fe[col], errors='coerce').fillna(0)
        else:
            df_fe[col] = 0 # Add column if missing
    
    # 3.1 Household structure & vulnerability
    df_fe['hh_size'] = df_fe['Family_Size']
    df_fe['dependents_count'] = df_fe['has_under5'] + df_fe['has_50_plus']
    df_fe['working_age_adults'] = df_fe['has_18_50_Family_member']
    df_fe['dependency_ratio'] = df_fe['dependents_count'] / (df_fe['working_age_adults'] + 1e-8)

    # 3.2 Economic capacity & liquidity
    df_fe['income_total_monthly'] = df_fe['Income_Monthly_per_head'] * df_fe['hh_size']
    df_fe['expenditure_total_monthly'] = df_fe['Income_Monthly_per_head'] * df_fe['hh_size']
    df_fe['income_pc'] = df_fe['Income_Monthly_per_head']
    df_fe['expenditure_pc'] = df_fe['expenditure_total_monthly'] / (df_fe['hh_size'] + 1e-8)
    df_fe['savings_to_expenditure'] = df_fe['Savings_Amt'] / (df_fe['expenditure_total_monthly'] + 1e-8)
    df_fe['debt_to_income'] = df_fe['Loans_Outstanding'] / (df_fe['Total_Income_Annualy'] + 1.0)
    df_fe['debt_to_income'] = df_fe['debt_to_income'].replace([float('inf'), float('-inf')], 0)
    df_fe['liquidity_gap'] = (df_fe['expenditure_total_monthly'] - df_fe['income_total_monthly']) / (df_fe['income_total_monthly'] + 1e-8)

    # 3.3 Assets & livelihood readiness
    df_fe['productive_asset_index'] = df_fe['Productive_Asset_Value']

    # 3.4 Health burden & care access
    df_fe['chronic_illness_count'] = df_fe['Chronic_Patient_Num']
    df_fe['disability_in_household'] = df_fe['Disabled_Yes']

    # 5. Encodings
    try:
        df_fe['Marrital_Status_Encoded'] = le_marital.transform(df_fe['Marrital_Status'].astype(str))
    except ValueError as e:
        logging.warning(f"Label encoder warning: {e}. Handling unknown labels.")
        known_classes = le_marital.classes_
        df_fe['Marrital_Status_Encoded'] = [cls if cls in known_classes else 'Others' for cls in df_fe['Marrital_Status'].astype(str)]
        try:
             df_fe['Marrital_Status_Encoded'] = le_marital.transform(df_fe['Marrital_Status_Encoded'])
        except Exception:
             df_fe['Marrital_Status_Encoded'] = 0 
    
    # 4. Aid-Type–Specific Scores
    scaler = StandardScaler()
    cash_need_features = ['liquidity_gap', 'debt_to_income']
    asset_readiness_features = ['productive_asset_index']
    health_need_features = ['chronic_illness_count', 'disability_in_household']
    training_suitability_features = ['Marrital_Status_Encoded']
    
    df_fe['cash_need_score'] = scaler.fit_transform(df_fe[cash_need_features].fillna(0)).mean(axis=1)
    df_fe['asset_readiness_score'] = scaler.fit_transform(df_fe[asset_readiness_features].fillna(0)).mean(axis=1)
    df_fe['health_need_score'] = scaler.fit_transform(df_fe[health_need_features].fillna(0)).mean(axis=1)
    df_fe['training_suitability_score'] = scaler.fit_transform(df_fe[training_suitability_features].fillna(0)).mean(axis=1)

    score_cols = ['cash_need_score', 'asset_readiness_score', 'health_need_score', 'training_suitability_score']
    score_df = df_fe[score_cols]
    
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-8)

    softmax_vals = softmax(score_df.values)
    for i, col in enumerate(score_cols):
        df_fe[f'{col}_softmax'] = softmax_vals[:, i]

    # 5. Interaction & Missingness
    df_fe['dependency_ratio_x_income_pc'] = df_fe['dependency_ratio'] * df_fe['income_pc']
    df_fe['productive_asset_x_market_access'] = df_fe['productive_asset_index'] * 0
    for col in ['income_pc', 'debt_to_income', 'savings_to_expenditure']:
        if col in df_fe.columns:
            df_fe[f'is_{col}_missing'] = df_fe[col].isna().astype(int)

    logging.info("Feature engineering complete.")
    return df_fe

# --- 3. Helper Function for SHAP ---
def get_top_drivers(shap_values_row, feature_names, top_n=3):
    """Extracts the top N features and their impact from a SHAP values row."""
    df = pd.DataFrame({'feature': feature_names, 'shap_value': shap_values_row})
    df['abs_shap'] = df['shap_value'].abs()
    df = df.sort_values(by='abs_shap', ascending=False).head(top_n)
    drivers = [{"feature": row['feature'], "contribution": round(row['shap_value'], 4)} for _, row in df.iterrows()]
    return drivers

# --- 4. Load All Models & Resources at Startup ---
try:
    logging.info("Loading models and resources...")
    # NOTE: Update these paths if your best models are different
    MODEL_T1_PATH = 'models/task1_GradientBoostingClassifier_calibrated_run_1.pkl'
    MODEL_T2_PATH = 'models/task2_GradientBoostingClassifier_with_eligibility_proba_run_1.pkl'
    
    model_t1 = joblib.load(MODEL_T1_PATH)
    model_t2 = joblib.load(MODEL_T2_PATH)
    le_marital = joblib.load('data/label_encoder_marital.pkl') 
    
    with open('data/final_features.json', 'r') as f:
        top_10_features = json.load(f) 

    logging.info("Creating SHAP explainers...")
    X_train_bg = pd.read_csv('data/processed_X_train.csv') 
    X_train_sample = shap.sample(X_train_bg, 50, random_state=42) 

    explainer_t1 = shap.Explainer(model_t1.predict_proba, X_train_sample)
    
    t1_probs_for_bg = model_t1.predict_proba(X_train_sample)[:, 1]
    X_train_sample_t2 = X_train_sample.copy()
    X_train_sample_t2['eligibility_proba'] = t1_probs_for_bg
    
    explainer_t2 = shap.Explainer(model_t2.predict_proba, X_train_sample_t2)
    logging.info("Models, resources, and explainers loaded successfully.")

except FileNotFoundError as e:
    logging.error(f"FATAL ERROR: Could not load model files. {e}")
    logging.error("Please ensure models and data files are in the correct 'models/' and 'data/' directories.")
    model_t1 = model_t2 = le_marital = top_10_features = explainer_t1 = explainer_t2 = None
except Exception as e:
    logging.error(f"FATAL ERROR during model loading: {e}")
    model_t1 = model_t2 = le_marital = top_10_features = explainer_t1 = explainer_t2 = None


# --- 5. The API Endpoints ---
@app.route("/")
def home():
    """Serves the frontend HTML file."""
    # This will look for 'index.html' in a folder named 'templates'
    # For simplicity, we'll just send the file from the current dir
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Performs the full inference pipeline."""
    if model_t1 is None:
        return jsonify({"error": "Models are not loaded. Server is not ready."}), 500

    try:
        raw_data = request.get_json()
        raw_df = pd.DataFrame([raw_data]) # Create a single-row DataFrame
        
        # 1. Engineer Features
        df_engineered = engineer_features_for_inference(raw_df, le_marital)
        
        # 2. Select Top 10 Features
        for col in top_10_features:
            if col not in df_engineered.columns:
                df_engineered[col] = 0
        X_new = df_engineered[top_10_features].fillna(0)
        
        # 3. Task 1: Eligibility Prediction
        task1_prob = model_t1.predict_proba(X_new)[:, 1][0] # [0] to get scalar
        
        # 4. Task 2: ML Aid Recommendation
        X_new_t2 = X_new.copy()
        X_new_t2['eligibility_proba'] = task1_prob
        task2_prob_all = model_t2.predict_proba(X_new_t2)
        task2_pred_idx = np.argmax(task2_prob_all[0])
        task2_pred_ml = model_t2.classes_[task2_pred_idx]
        
        # 5. SHAP Explanations
        shap_values_t1_all = explainer_t1(X_new)
        shap_values_t2_all = explainer_t2(X_new_t2)
        
        top_3_t1 = get_top_drivers(shap_values_t1_all[0, :, 1].values, X_new.columns)
        top_3_t2 = get_top_drivers(shap_values_t2_all[0, :, task2_pred_idx].values, X_new_t2.columns)
        
        # 6. Apply Business Logic & Mapping
        health_score = df_engineered.iloc[0]['health_need_score_softmax']
        cash_score = df_engineered.iloc[0]['cash_need_score']
        
        # (Tune these thresholds based on your data)
        HEALTH_NEED_THRESHOLD = 0.5 
        CASH_NEED_THRESHOLD = 0.0   

        final_aid = "N/A"
        if health_score > HEALTH_NEED_THRESHOLD:
            final_aid = 'Health Support'
        elif cash_score > CASH_NEED_THRESHOLD:
            final_aid = 'Cash Grant'
        elif task2_pred_ml == 'Enterprise Development':
            final_aid = 'Livelihood Asset'
        elif task2_pred_ml == 'Skill Development':
            final_aid = 'Training'
        elif task2_pred_ml == 'Both':
            final_aid = 'Livelihood Asset AND Training'

        # 7. Format JSON Response
        response = {
            "participant_id": raw_data.get('Participant_ID', 'N/A'),
            "aid_eligibility_probability": round(task1_prob, 4),
            "top_3_eligibility_drivers": top_3_t1,
            "recommended_aid": final_aid,
            "top_3_recommendation_drivers": top_3_t2
        }
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

# --- 6. Run the App ---
if __name__ == "__main__":
    # Create a 'templates' folder if it doesn't exist for Flask
    if not os.path.exists('templates'):
        os.makedirs('templates')

    
    # Let's use a simpler setup: serve 'index.html' from the root
    @app.route('/')
    def index():
        return app.send_static_file('index.html')

    app.run(host="0.0.0.0", port=5000, debug=False)
