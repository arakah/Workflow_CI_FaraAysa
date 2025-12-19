import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# KITA KOSONGKAN URI AGAR TERSIMPAN DI LAPTOP (LOKAL)
mlflow.set_tracking_uri("") 
mlflow.set_experiment("Serving_Lokal_Proof")

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"✅ RUN ID LOKAL BARU: {run_id}")
    
    # Bikin data & model dummy
    X, y = make_classification(n_samples=50)
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Simpan dengan nama 'model_rf_tuned' biar namanya sama kayak command kamu
    mlflow.sklearn.log_model(model, "model_rf_tuned")
    
    print("✅ Model lokal berhasil dibuat. Silakan copy Run ID di atas!")