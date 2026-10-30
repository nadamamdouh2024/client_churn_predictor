# 📊 Telco Customer Churn Predictor

An end-to-end Machine Learning project designed to predict customer churn in the telecommunications sector. This project automates data preprocessing, trains and evaluates multiple classification models using **Scikit-Learn Pipelines**, saves the best-performing model, and provides an interactive **Streamlit** web application for real-time predictions.

---

## 🚀 Key Features

* **Automated Data Preprocessing (`dataprocessing.py`):**
  * Cleans raw data, removes unnecessary identifiers (`customerID`), and converts `TotalCharges` to numeric format.
  * Normalizes binary features (`SeniorCitizen`) and encodes the target variable (`Churn`).
  * Implements a robust `ColumnTransformer` pipeline:
    * **Numerical Pipeline:** Median imputation (`SimpleImputer`) + Standardization (`StandardScaler`).
    * **Categorical Pipeline:** Most-frequent imputation (`SimpleImputer`) + One-Hot Encoding (`OneHotEncoder`).
* **Multi-Model Training & Benchmarking (`train.py`):**
  * Trains and compares **6 Machine Learning classifiers** wrapped in end-to-end pipelines to prevent data leakage.
  * Evaluates models across 5 key metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.
  * Automatically selects and exports the highest-performing model (`churn_model.joblib`) alongside a comprehensive evaluation report (`results.csv`).
* **Interactive Web Interface:**
  * Built with **Streamlit** for seamless user interaction and instant customer churn probability estimation.

---

## 🧠 Machine Learning Models Evaluated

The training script benchmarks the following algorithms with customized hyperparameters:

1. **Logistic Regression** (`C=10`, `max_iter=3000`)
2. **K-Nearest Neighbors (KNN)** (`n_neighbors=21`, Manhattan distance `p=1`)
3. **Gaussian Naive Bayes**
4. **Decision Tree Classifier** (`max_depth=5`, `criterion='gini'`)
5. **Random Forest Classifier** (`n_estimators=200`, `max_depth=15`, `criterion='entropy'`)
6. **Support Vector Machine (SVM)** (`kernel='rbf'`, `C=1`, `gamma=0.1`)

---

## 📂 Project Structure

```text
client_churn_predictor/
│
├── Data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Telco Customer Churn Dataset
│
├── models/                                     # Generated automatically after training
│   ├── churn_model.joblib                      # Best-performing serialized Pipeline
│   ├── results.csv                             # Evaluation metrics for all 6 models
│   └── best_model_name.txt                     # Name of the winning model
│
├── dataprocessing.py                           # Data loading, cleaning & ColumnTransformer pipeline
├── train.py                                    # Model training, evaluation & serialization script
├── app.py                                      # Streamlit web application (if applicable)
├── requirements.txt                            # Project dependencies
└── README.md                                   # Project documentation
```
---

## 📊 Dataset Features

The model uses the **Telco Customer Churn** dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), utilizing **19 input features** to predict the target variable (`Churn`):

* **Numerical Features (3):** `tenure`, `MonthlyCharges`, `TotalCharges`
* **Categorical Features (16):** `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`
* **Target Variable:** `Churn` (`Yes` -> `1`, `No` -> `0`)

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/nadamamdouh2024/client_churn_predictor.git](https://github.com/nadamamdouh2024/client_churn_predictor.git)
cd client_churn_predictor
```

### 2. Install Dependencies
Make sure you have Python installed, then install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Train the Models
Run the training script to preprocess the data, evaluate all models, and generate the `models/` artifacts:
```bash
python train.py
```

### 4. Run the Streamlit App
Launch the interactive web application:
```bash
streamlit run app.py
```
*(Note: Replace `app.py` with the exact filename of your Streamlit script if named differently).*

---

## 🛠️ Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Joblib
* **Web Framework:** Streamlit

---

## 👩‍💻 Author

**Nada Mamdouh**
* **GitHub:** [@nadamamdouh2024](https://github.com/nadamamdouh2024)





