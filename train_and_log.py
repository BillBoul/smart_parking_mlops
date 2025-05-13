import argparse
import mlflow
import mlflow.keras
import numpy as np
import matplotlib.pyplot as plt
import os
from data_loader import load_dataset
from feature_engineering import add_time_features, prepare_features_and_labels, split_and_scale
from model_builder import build_model
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
import shutil
import urllib.parse
import joblib

def main(args):
    # ==== Load and preprocess data ====
    df = load_dataset(args.csv_path)
    df = add_time_features(df)
    X, y = prepare_features_and_labels(df)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(X, y, return_scaler=True)

    # ==== Build and train model ====
    input_dim = X_train_scaled.shape[1]
    model = build_model(input_dim)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=f"Run_{args.epochs}ep_{args.batch_size}bs"):
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("model_type", "DNN with Huber and L2")

        history = model.fit(
            X_train_scaled, y_train,
            validation_split=args.validation_split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[early_stop],
            verbose=1
        )

        # ==== Evaluate ====
        y_pred = model.predict(X_test_scaled).flatten()
        y_pred_rounded = np.clip(np.round(y_pred), 0, 8)

        mae = mean_absolute_error(y_test, y_pred_rounded)
        r2 = r2_score(y_test, y_pred_rounded)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # ==== Log model ====
        mlflow.keras.log_model(model, artifact_path="final_model")
        print("✅ Model logged to MLflow.")

        # Get actual model path on disk from MLflow URI
        artifact_uri = mlflow.get_artifact_uri("final_model")  # e.g., file:///C:/Users/...
        parsed_uri = urllib.parse.urlparse(artifact_uri)
        local_model_path = parsed_uri.path

        # Windows fix: remove leading slash (e.g., '/C:/...' → 'C:/...')
        if os.name == "nt" and local_model_path.startswith("/"):
            local_model_path = local_model_path[1:]

        print(f"[INFO] Resolved model path: {local_model_path}")

        production_model_path = "production_model"

        if os.path.exists(production_model_path):
            shutil.rmtree(production_model_path)

        shutil.copytree(local_model_path, production_model_path)
        print(f"[INFO] Best model copied to '{production_model_path}' for API deployment.")

        scaler_path = os.path.join("production_model", "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)
        print(f"[INFO] Scaler saved to {scaler_path} and logged to MLflow.")

        # ==== Plot and log prediction results ====
        os.makedirs("output", exist_ok=True)
        plt.figure(figsize=(12, 6))
        plt.plot(range(100), y_test[:100], label='True Available Spots', marker='o')
        plt.plot(range(100), y_pred_rounded[:100], label='Predicted Available Spots', marker='x')
        plt.title('Validation Set: True vs Predicted Parking Availability (First 100 Samples)')
        plt.xlabel('Sample Index')
        plt.ylabel('Available Spots')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = "output/actual_vs_predicted_first100.png"
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path)
        print(f"Plot logged to MLflow: {plot_path}")

        # ==== Plot and log training & validation loss ====
        loss_plot_path = "output/training_validation_loss.png"
        plt.figure(figsize=(10, 6))
        plt.plot(history.history["loss"], label="Training Loss", marker='o')
        plt.plot(history.history["val_loss"], label="Validation Loss", marker='x')
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(loss_plot_path)
        plt.close()

        mlflow.log_artifact(loss_plot_path)
        print(f"Loss curve logged to MLflow: {loss_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and log a Smart Parking DNN model.")
    parser.add_argument("--csv_path", type=str, default="data/merged_parking_weather_dataset.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--experiment_name", type=str, default="Smart Parking Prediction",
                        help="MLflow experiment name")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="Fraction of training data to use for validation")

    args = parser.parse_args()
    main(args)
