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

# --- KONFIGURASI DAGSHUB ---
DAGSHUB_USER = "aysafara.04"
REPO_NAME = "submission-mlops"
dagshub.init(repo_owner=DAGSHUB_USER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{REPO_NAME}.mlflow")

# Set Experiment Khusus Skilled
mlflow.set_experiment("Eksperimen_Skilled_Tuning")

def train_skilled():
    print("[INFO] Memulai Training Skilled (Manual Tuning)...")
    
    # 1. Load Data
    if os.path.exists("loan_data_cleaned_automated.csv"):
        df = pd.read_csv("loan_data_cleaned_automated.csv")
    elif os.path.exists("namadataset_preprocessing/loan_data_cleaned_automated.csv"):
         df = pd.read_csv("namadataset_preprocessing/loan_data_cleaned_automated.csv")
    else:
        print("Error: Dataset tidak ditemukan.")
        return

    target_col = 'loan_status'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Hyperparameter Tuning (Grid Search)
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
    }
    
    # Verbose=0 biar terminal ga penuh
    grid = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', verbose=1)
    
    # Start Run Manual
    with mlflow.start_run(run_name="Manual_Tuning_Run"):
        print("Sedang melakukan GridSearch...")
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        
        y_pred = best_model.predict(X_test)
        
        # Hitung Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # --- LOGGING MANUAL (Syarat Skilled) ---
        print("Logging data ke MLflow secara manual...")
        
        # Log Hyperparameters
        mlflow.log_params(best_params)
        
        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # Log Model
        mlflow.sklearn.log_model(best_model, "model_rf_tuned")
        
        # Log Artifact (Gambar)
        plt.figure(figsize=(6,4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix (Manual Log)")
        plt.savefig("confusion_matrix_skilled.png")
        mlflow.log_artifact("confusion_matrix_skilled.png")
        
        print("Selesai! Cek DagsHub.")

if __name__ == "__main__":
    train_skilled()