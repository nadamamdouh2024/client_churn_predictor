import joblib 
import pandas as pd 
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from dataprocessing import build_preprocessor,load_data,split_xy

BASE_DIR=Path(__file__).parent
MODELS_DIR=BASE_DIR/"models"
MODELS_DIR.mkdir(exist_ok=True)

df= load_data()
X,Y=split_xy(df)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)



models = {
    "logistic regression":LogisticRegression(max_iter=3000,class_weight=None,random_state=42,C=10,l1_ratio=0),
    "knn":KNeighborsClassifier(n_neighbors=21,p=1,weights="uniform",metric="minkowski"),
    "naive bayes":GaussianNB(),
    "decision tree":DecisionTreeClassifier(max_depth=5,class_weight=None,random_state=42,min_samples_split=20,min_samples_leaf=4,max_features=None,criterion="gini"),
    "random forest":RandomForestClassifier(n_estimators=200,class_weight=None,min_samples_split=2,min_samples_leaf=8,max_features="sqrt",max_depth=15,criterion="entropy",bootstrap=True,random_state=42),
    "svm":SVC(class_weight=None,gamma=0.1,kernel="rbf",C=1,random_state=42,probability=True)
}

results=[]
trained={}

for name,clf in models.items():
    pipe =Pipeline([
        ("processor",build_preprocessor()),
        ("models",clf)
    ])
    pipe.fit(X_train,Y_train)
    predictions = pipe.predict(X_test)
    probabilities=pipe.predict_proba(X_test)[:,1]   #print probability of churn(1)

    results.append({
        "Model":name,
        "Accuracy":accuracy_score(Y_test,predictions),
        "Precision":precision_score(Y_test,predictions),
        "Recall":recall_score(Y_test,predictions),
        "F1":f1_score(Y_test,predictions),
        "ROC_AUC":roc_auc_score(Y_test,probabilities)
    })


    trained[name]=pipe    #save الموديل with ال بروسيس بتاعه
    print(f"done  {name}")



results_df = pd.DataFrame(results).sort_values(
    "Accuracy", ascending=False
).reset_index(drop=True)
print("\n", results_df.round(3).to_string(index=False))
best_name = results_df.loc[0, "Model"]
print(f"\nBest model (by Accuracy): {best_name}")


results_df.to_csv(MODELS_DIR / "results.csv", index=False)
joblib.dump(trained[best_name], MODELS_DIR / "churn_model.joblib")
(MODELS_DIR / "best_model_name.txt").write_text(best_name)


import sklearn
import sys

print(sys.version)
print(sklearn.__version__)
