Okay, here is the plan for the project, including the technical documentation (`Documentation.md`) and the project structure.

---

## `Documentation.md`

# AI/ML Model for Poverty Graduation Support

## 1. Project Overview

### 1.1 Background

A social development programme aims to support households escaping extreme poverty through various interventions (assets, training, healthcare). The goal is to leverage AI/ML to automate parts of the eligibility assessment and aid recommendation process.

### 1.2 Objective

Develop an AI/ML model to:

1.  Predict the probability (percentage) that a household qualifies for Income-Generating Activity (IGA) support (Task 1).
2.  Recommend the most suitable aid type (e.g., Cash Grant, Livelihood Asset, Health Support, Training) for poverty graduation (Task 2).

### 1.3 Data Source

AI/ML Model for Poverty Graduation Support1. Project Overview1.1 BackgroundA social development programme aims to support households escaping extreme poverty through various interventions (assets, training, healthcare). The goal is to leverage AI/ML to automate parts of the eligibility assessment and aid recommendation process.1.2 ObjectiveDevelop an AI/ML model to:Predict the probability (percentage) that a household qualifies for Income-Generating Activity (IGA) support (Task 1).Recommend the most suitable aid type (e.g., Cash Grant, Livelihood Asset, Health Support, Training) for poverty graduation (Task 2).1.3 Data SourceA CSV file (Participant_Selection_Final.csv) containing demographic, income, health, and asset information for participants.2. Technical Approach2.1 Data LoadingUse pandas to load the Participant_Selection_Final.csv file.2.2 Feature EngineeringData Cleaning:Handle missing values (e.g., imputation, dropping).Identify and treat outliers if necessary.Check data types and convert if needed.Feature Creation:Derive new features from existing ones (e.g., age from DOB, income per capita).Engineer categorical features (e.g., encoding marital status).Feature Selection:Identify features most relevant to the tasks.Use statistical methods (e.g., correlation) or model-based feature importance.Select the most important features for modeling.2.3 Exploratory Data Analysis (EDA)Perform EDA on the selected important features.Visualize distributions, relationships, and correlations.Generate plots (e.g., histograms, scatter plots, heatmaps) and save them as image files.2.4 Modeling2.4.1 Task 1: IGA Eligibility Probability PredictionProblem Type: Binary Classification (Eligible/Not Eligible for IGA) or Regression (Probability Score).Models: Use 4 Scikit-learn ensemble models (e.g., RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier).Training:Split the data into training and validation sets (e.g., 80/20).Train each model on the training set.Potentially tune hyperparameters using cross-validation (e.g., GridSearchCV).Evaluation:Use appropriate metrics for classification/regression (e.g., Accuracy, Precision, Recall, F1-Score, ROC-AUC for classification; MSE, MAE, R2 for regression).Compare model performance.Model Saving:Save the best performing model using pickle or joblib.Save evaluation results (metrics, reports) to a file.2.4.2 Task 2: Aid Type RecommendationProblem Type: Multi-class Classification (predicting the recommended aid type).Models: Use 4 Scikit-learn ensemble models (e.g., RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier).Training:Use the same training/validation split or a different one if appropriate.Train each model on the training set to predict the aid type.Potentially tune hyperparameters using cross-validation.Evaluation:Use appropriate metrics for multi-class classification (e.g., Accuracy, Precision, Recall, F1-Score per class, Overall F1-Score, Confusion Matrix).Compare model performance.Model Saving:Save the best performing model using pickle or joblib.Save evaluation results (metrics, reports) to a file.2.5 Test InferenceLoad the saved models.Prepare a small sample of unseen data (or use the test set from the split).Run inference using the loaded models for both Task 1 and Task 2.Display the results (predicted probabilities/classes).3. Tools & LibrariesPython 3.xpandasnumpyscikit-learnmatplotlib / seaborn (for EDA)pickle / joblib (for model saving)4. Project Structure(See Project Structure section below)Project Structurepoverty_graduation_ml/
│
├── README.md # (Optional) Brief overview of the project
├── Documentation.md # This file
├── data/
│ └── Participant_Selection_Final.csv # The raw input data file
├── src/
│ ├── data_loading.py # Script for loading the dataset
│ ├── feature_engineering.py # Script for cleaning, creating, and selecting features
│ ├── eda.py # Script for performing EDA and saving plots
│ ├── model_training_task1.py # Script for training models for Task 1
│ ├── model_training_task2.py # Script for training models for Task 2
│ └── inference.py # Script for loading models and running test inference
├── models/ # Directory to save trained models (created by scripts)
│ ├── task1_best_model.pkl
│ ├── task2_best_model.pkl
│ └── ...
├── results/ # Directory to save evaluation results, metrics, reports (created by scripts)
│ ├── task1_evaluation.txt
│ ├── task2_evaluation.txt
│ └── ...
├── plots/ # Directory to save EDA plots (created by scripts)
│ ├── feature_importance.png
│ ├── income_distribution.png
│ └── ...
└── notebook.ipynb # Jupyter notebook orchestrating the pipeline by calling functions from `src/`
This structure keeps the code organized, separates the logic into distinct scripts, and allows the notebook to serve as the main orchestrator and documentation hub for the workflow.
A CSV file (`Participant_Selection_Final.csv`) containing demographic, income, health, and asset information for participants.

## 2. Technical Approach

### 2.1 Data Loading

- Use `pandas` to load the `Participant_Selection_Final.csv` file.

### 2.2 Feature Engineering

- **Data Cleaning:**
  - Handle missing values (e.g., imputation, dropping).
  - Identify and treat outliers if necessary.
  - Check data types and convert if needed.
- **Feature Creation:**
  - Derive new features from existing ones (e.g., age from DOB, income per capita).
  - Engineer categorical features (e.g., encoding marital status).
- **Feature Selection:**
  - Identify features most relevant to the tasks.
  - Use statistical methods (e.g., correlation) or model-based feature importance.
  - Select the most important features for modeling.

### 2.3 Exploratory Data Analysis (EDA)

- Perform EDA on the selected important features.
- Visualize distributions, relationships, and correlations.
- Generate plots (e.g., histograms, scatter plots, heatmaps) and save them as image files.

### 2.4 Modeling

#### 2.4.1 Task 1: IGA Eligibility Probability Prediction

- **Problem Type:** Binary Classification (Eligible/Not Eligible for IGA) or Regression (Probability Score).
- **Models:** Use 4 Scikit-learn ensemble models (e.g., RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier).
- **Training:**
  - Split the data into training and validation sets (e.g., 80/20).
  - Train each model on the training set.
  - Potentially tune hyperparameters using cross-validation (e.g., GridSearchCV).
- **Evaluation:**
  - Use appropriate metrics for classification/regression (e.g., Accuracy, Precision, Recall, F1-Score, ROC-AUC for classification; MSE, MAE, R2 for regression).
  - Compare model performance.
- **Model Saving:**
  - Save the best performing model using `pickle` or `joblib`.
  - Save evaluation results (metrics, reports) to a file.

#### 2.4.2 Task 2: Aid Type Recommendation

- **Problem Type:** Multi-class Classification (predicting the recommended aid type).
- **Models:** Use 4 Scikit-learn ensemble models (e.g., RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier).
- **Training:**
  - Use the same training/validation split or a different one if appropriate.
  - Train each model on the training set to predict the aid type.
  - Potentially tune hyperparameters using cross-validation.
- **Evaluation:**
  - Use appropriate metrics for multi-class classification (e.g., Accuracy, Precision, Recall, F1-Score per class, Overall F1-Score, Confusion Matrix).
  - Compare model performance.
- **Model Saving:**
  - Save the best performing model using `pickle` or `joblib`.
  - Save evaluation results (metrics, reports) to a file.

### 2.5 Test Inference

- Load the saved models.
- Prepare a small sample of unseen data (or use the test set from the split).
- Run inference using the loaded models for both Task 1 and Task 2.
- Display the results (predicted probabilities/classes).

## 3. Tools & Libraries

- Python 3.x
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib` / `seaborn` (for EDA)
- `pickle` / `joblib` (for model saving)

## 4. Project Structure

(See `Project Structure` section below)

---

## Project Structure

```
poverty_graduation_ml/
│
├── README.md                 # (Optional) Brief overview of the project
├── Documentation.md          # This file
├── data/
│   └── Participant_Selection_Final.csv  # The raw input data file
├── src/
│   ├── data_loading.py       # Script for loading the dataset
│   ├── feature_engineering.py # Script for cleaning, creating, and selecting features
│   ├── eda.py               # Script for performing EDA and saving plots
│   ├── model_training_task1.py # Script for training models for Task 1
│   ├── model_training_task2.py # Script for training models for Task 2
│   └── inference.py         # Script for loading models and running test inference
├── models/                  # Directory to save trained models (created by scripts)
│   ├── task1_best_model.pkl
│   ├── task2_best_model.pkl
│   └── ...
├── results/                 # Directory to save evaluation results, metrics, reports (created by scripts)
│   ├── task1_evaluation.txt
│   ├── task2_evaluation.txt
│   └── ...
├── plots/                   # Directory to save EDA plots (created by scripts)
│   ├── feature_importance.png
│   ├── income_distribution.png
│   └── ...
└── notebook.ipynb           # Jupyter notebook orchestrating the pipeline by calling functions from `src/`
```

This structure keeps the code organized, separates the logic into distinct scripts, and allows the notebook to serve as the main orchestrator and documentation hub for the workflow.
