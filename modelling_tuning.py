import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DAGSHUB_USER = os.getenv("DAGSHUB_USERNAME", "aysafara.04")
REPO_NAME = "submission-mlops" 
DATA_PATH = "data/loan_data_cleaned_automated.csv" 


try:
    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=REPO_NAME, mlflow=True)
except:
    print("DagsHub init warning (mungkin sudah ter-init atau environment variables belum set).")

mlflow.set_experiment("Eksperimen_Loan_Prediction_Advance")

def main():
    print(f"[INFO] Loading data: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File {DATA_PATH} tidak ditemukan.")

    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing simple (Pastikan target sesuai dataset)
    target_col = 'loan_status' 
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter Tuning
    print("[INFO] Starting Grid Search...")
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy')
    
    # Start MLflow Run
    with mlflow.start_run(run_name="CI_CD_Run_Tuned") as run:
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_

        # Prediksi
        y_pred = best_model.predict(X_test)
        
        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1": f1_score(y_test, y_pred, average='weighted')
        }

        # --- MANUAL LOGGING ---
        print("[INFO] Logging to DagsHub...")
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "model")

        # --- ARTEFAK VISUAL ---
        # 1. Confusion Matrix
        plt.figure(figsize=(8,6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # 2. Feature Importance
        if hasattr(best_model, 'feature_importances_'):
            plt.figure(figsize=(10,6))
            pd.Series(best_model.feature_importances_, index=X.columns).nlargest(10).plot(kind='barh')
            plt.title("Feature Importance")
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            
        print(f"[SUCCESS] Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()