Here is a well-structured `README.md` for your project:

# AI/ML Model for Aid Recommendation

## 1. Project Overview

This project develops a two-part machine learning pipeline to support a social development program. The objectives are:

- **Task 1: Eligibility Prediction**: Predict the probability (%) that a household qualifies for Income-Generating Activity (IGA) support.
- **Task 2: Aid Recommendation**: Recommend the most suitable aid type (e.g., Cash Grant, Livelihood Asset, Health Support, Training) to help the household move out of extreme poverty.

The project includes all scripts for feature engineering, training, and evaluation, orchestrated by a central Jupyter Notebook. It also includes a Flask web application that serves a form for field officers to get real-time predictions and aid recommendations for new households.

## 2. Project Structure

```

poverty_graduation_ml/
│
├── app.py                   # Flask API server for the web application
├── notebook.ipynb           # Main Jupyter Notebook for running the E2E ML pipeline
├── requirements.txt         # All Python dependencies
├── Documentation.md         # Detailed project documentation
├── templates/
│   └── index.html           # HTML/CSS/JS frontend for the web application
│
├── data/
│   ├── Participant_Selection_Final.csv  # Raw input data
│   ├── processed_X_train.csv  # Processed data (created by notebook)
│   ├── final_features.json    # List of selected features (created by notebook)
│   └── ...
│
├── src/
│   ├── feature_engineering.py # Script for cleaning, creating, and selecting features
│   ├── exploratory_data_analysis.py # Script for EDA
│   ├── model_training_task1.py # Script for training Task 1 models
│   ├── model_training_task2.py # Script for training Task 2 models
│   └── inference.py         # Script for inference logic (used by notebook and app)
│
├── models/                  # Directory for saved models (created by notebook)
│   ├── task1_... .pkl
│   └── task2_... .pkl
│
├── results/                 # Directory for evaluation CSVs (created by notebook)
│   ├── task1_evaluation_... .csv
│   └── task2_evaluation_... .csv
│
└── plots/                   # Directory for saved EDA plots (created by notebook)
├── eda_... .png
└── ...

```

## 3. Setup & Installation

### Clone the Repository

Clone the repository (if applicable):

```bash
git clone [your-repo-url]
cd poverty_graduation_ml
```

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install Dependencies

Ensure you have `requirements.txt` (as provided in our conversation) in your root directory.

```bash
pip install -r requirements.txt
```

## 4. How to Run: Model Training Pipeline

The entire ML pipeline (feature engineering, training, evaluation) is orchestrated in `notebook.ipynb`. Run this first to generate all the necessary models and files.

### Start Jupyter

```bash
jupyter notebook
```

### Open `notebook.ipynb`

Open the notebook in your browser.

### Run All Cells

Run all cells sequentially from top to bottom. This will:

- Load the raw data from `data/Participant_Selection_Final.csv`.
- Run feature engineering and save the processed datasets (`processed_X_train.csv`, etc.) and `final_features.json` to the `data/` folder.
- Save the `label_encoder_marital.pkl` to the `data/` folder.
- Run and evaluate the Task 1 models, saving the best one to the `models/` folder and results to the `results/` folder.
- Run and evaluate the Task 2 models, saving the best one to the `models/` folder and results to the `results/` folder.
- Generate EDA and result visualization plots.

## 5. How to Run: Web Application (API & Frontend)

After you have run the notebook at least once, you can start the web application.

### Ensure `index.html` is in the Templates Folder

- Create a folder named `templates` in the root of your project (if it doesn't exist).
- Move the `index.html` file into this `templates` folder. Flask automatically looks for HTML files here.

### Start the Flask Server

In your terminal (from the project's root directory), run:

```bash
python app.py
```

You should see output indicating the server is running on `http://127.0.0.1:5000/`. The server will also log that it has successfully loaded all models and SHAP explainers.

### Use the Application

1. Open your web browser and go to: [http://127.0.0.1:5000/](http://127.0.0.1:5000/).
2. The "Poverty Graduation Aid Advisor" form will load.
3. Fill out the household data. The form is pre-filled with sample data for easy testing.
4. Click "Get Recommendation."
5. The frontend will call the API, and the results (Eligibility Score, Recommended Aid, and Top 3 Drivers) will appear on the page.

---

This README provides an overview of how to set up, train, and use the machine learning models and Flask web application. If you have any issues or questions, refer to the `Documentation.md` for detailed instructions on the project setup and usage.
