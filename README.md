# ECG Heartbeat Classification Project

This project focuses on the analysis, modeling, and classification of ECG (Electrocardiogram) heartbeat signals using machine learning techniques. The primary goal is to accurately categorize heartbeats into different conditions using data-driven approaches.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- I[nstallation & Setup](#installation--setup)
- [Usage](#usage)
- [Modeling Approach](#modeling-approach)
- [Results & Evaluation](#results--evaluation)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project leverages various machine learning models, including XGBoost, LightGBM, and CatBoost, to classify ECG heartbeats. The workflow includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and visualization of results.

---

## Dataset

- **Source:** [ECG Heartbeat Categorization Dataset on Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data?select=mitbih_train.csv)
- **Description:** Each row represents a single heartbeat captured as an ECG signal, with features extracted from the signal and a label indicating the heartbeat class.

---

## Project Structure

```
data/
    ecg_test.csv
    ecg_train.csv
    heartbeat/
        mitbih_test.csv
        mitbih_train.csv
        ptbdb_abnormal.csv
        ptbdb_normal.csv
models/
    heartbeat_model.pkl
notebooks/
    analysis and modeling.ipynb
```

- **data/heartbeat/**: Contains the raw and processed dataset files.
- **models/**: Stores trained model artifacts (e.g., `heartbeat_model.pkl`).
- **notebooks/**: Jupyter notebooks for data analysis, modeling, and visualization.

---

## Installation & Setup

1. **Clone the repository:**

   ```powershell
   git clone https://github.com/mhmdkardosha/Electronics-project-ECG-classifier.git
   cd "Electronics project"
   ```

2. **Create and activate a virtual environment:**

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

   *(If `requirements.txt` is missing, install the following manually: pandas, numpy, scikit-learn, xgboost, lightgbm, catboost, matplotlib, seaborn, joblib, kagglehub)*

4. **Download the dataset:**
   - The notebook uses `kagglehub` to download the dataset automatically if not present.

---

## Usage

1. **Run the Jupyter Notebook:**

   ```powershell
   jupyter notebook "notebooks/analysis and modeling.ipynb"
   ```

2. **Notebook Workflow:**
   - **Data Download & Preparation:** Downloads and moves the dataset to the appropriate directory.
   - **EDA:** Visualizes and explores the dataset.
   - **Model Training:** Trains models (XGBoost, LightGBM, CatBoost) with sample weighting.
   - **Evaluation:** Prints classification reports and confusion matrices.
   - **Model Saving:** Saves the best model as `heartbeat_model.pkl`.
   - **Model Loading & Testing:** Loads the saved model and evaluates on test data.

---

## Modeling Approach

- **Feature Engineering:** Uses raw features from the dataset.
- **Class Imbalance Handling:** Applies sample weighting during model training.
- **Models Used:**
  - XGBoost (`XGBClassifier`)
  - LightGBM (`LGBMClassifier`)
  - CatBoost (`CatBoostClassifier`)
- **Evaluation Metrics:** Precision, recall, F1-score, confusion matrix.

---

## Results & Evaluation

- **Best Model:** XGBoost classifier, saved as [`models/heartbeat_model.pkl`](models/heartbeat_model.pkl ).
- **Performance:** Detailed classification reports and confusion matrices are generated in the notebook.
- **Visualization:** Includes heatmaps and other plots for model evaluation.

---

## Future Work

- **Consult Medical Experts:** To enhance feature engineering and interpretation.
- **Advanced EDA:** More visualizations and statistical analyses.
- **Model Optimization:** Hyperparameter tuning and ensemble methods.
- **Deployment:** Building an API or web interface for real-time predictions.

---

## Acknowledgements

- Dataset by [Shayan Fazeli](https://www.kaggle.com/shayanfazeli) on Kaggle.
- Project inspired and developed by Mohamed Kardosha.

---

**Thank you for exploring this project! If you find it useful, please consider giving it a star.**

---

*For any questions or contributions, feel free to open an issue or pull request.*
