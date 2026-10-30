import pathlib as path
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
BASE_DIR= Path(__file__).parent
DATA_PATH=BASE_DIR/"Data"/"WA_Fn-UseC_-Telco-Customer-Churn.csv"


TARGET = "Churn"

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

CAT_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


def load_data(path=DATA_PATH):
    df=pd.read_csv(path)

    df=df.drop(columns=["customerID"],errors="ignore")

    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
    if df["SeniorCitizen"].dtype != object :
        df["SeniorCitizen"]=df["SeniorCitizen"].map({0:"No",1:"Yes"})

    df[TARGET]=df[TARGET].map({"Yes":1,"No":0}) 
    return df


def split_xy(df):
    X=df.drop(TARGET,axis=1)
    y=df[TARGET]
    return X,y

def build_preprocessor():
    num_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])
    cat_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num_pipeline, NUM_COLS),
        ("cat", cat_pipeline, CAT_COLS)
    ])


