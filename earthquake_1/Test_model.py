import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import sys
import seaborn as sns
import matplotlib.pyplot as plt

# --- 0. Configuration ---
DATA_FILE = 'earthquake_alert_data.csv' # The original data file
FEATURE_COLUMNS = ['magnitude', 'cdi', 'mmi'] # Must match training
TARGET_COLUMN = 'alert'                    # Must match training
MODEL_FILE_NAME = 'alert_classifier_model.keras'
SCALER_FILE_NAME = 'alert_scaler.joblib'
LABEL_ENCODER_FILE_NAME = 'alert_label_encoder.joblib'

# --- 1. Load Data ---
print(f"Loading data from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)
    # Drop rows with missing values in relevant columns (same as in training)
    df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN], inplace=True)
    if df.empty:
        print(f"Error: No valid data found in {DATA_FILE} after dropping missing values.")
        sys.exit()
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found.")
    sys.exit()
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit()

print(f"Loaded {len(df)} rows for evaluation.")

# Separate features (X) and target text labels (y_text)
X_eval = df[FEATURE_COLUMNS].astype(np.float32)
y_text_eval = df[TARGET_COLUMN]

# --- 2. Load Saved Model, Scaler, and Label Encoder ---
print("\nLoading saved model, scaler, and label encoder...")
try:
    model = load_model(MODEL_FILE_NAME)
    scaler = joblib.load(SCALER_FILE_NAME)
    label_encoder = joblib.load(LABEL_ENCODER_FILE_NAME)
    print("Model, scaler, and label encoder loaded successfully.")
except FileNotFoundError:
    print(f"Error: Could not find required files ('{MODEL_FILE_NAME}', '{SCALER_FILE_NAME}', '{LABEL_ENCODER_FILE_NAME}').")
    print("Please make sure you have run train_alert_classifier.py successfully first.")
    sys.exit()
except Exception as e:
    print(f"Error loading files: {e}")
    sys.exit()

# --- 3. Preprocess Evaluation Data ---
print("\nPreprocessing evaluation data...")
try:
    # Scale the features using the loaded scaler
    X_eval_scaled = scaler.transform(X_eval)

    # Encode the true text labels using the loaded label encoder
    y_eval_encoded = label_encoder.transform(y_text_eval)
    num_classes = len(label_encoder.classes_)

except Exception as e:
    print(f"Error during preprocessing: {e}")
    sys.exit()

# --- 4. Make Predictions ---
print("\nMaking predictions on the evaluation data...")
try:
    # Predict probabilities for each class
    y_pred_probs = model.predict(X_eval_scaled)
    # Get the class index with the highest probability
    y_pred_encoded = np.argmax(y_pred_probs, axis=1)

except Exception as e:
    print(f"An error occurred during prediction: {e}")
    sys.exit()

# --- 5. Evaluate Predictions ---
print("\n--- Model Evaluation Results ---")
try:
    accuracy = accuracy_score(y_eval_encoded, y_pred_encoded)
    print(f"Overall Accuracy on the dataset: {accuracy*100:.2f}%")

    print("\nClassification Report:")
    target_names = label_encoder.classes_
    # Ensure report uses labels present in the actual evaluation data
    unique_eval_labels = np.unique(y_eval_encoded)
    present_labels = np.intersect1d(unique_eval_labels, range(num_classes)) # Ensure labels are valid indices
    present_target_names = [target_names[i] for i in present_labels]

    print(classification_report(y_eval_encoded, y_pred_encoded, labels=present_labels, target_names=present_target_names, zero_division=0))

    print("\nConfusion Matrix:")
    # Calculate matrix based on present labels for better visualization if some classes are missing
    cm = confusion_matrix(y_eval_encoded, y_pred_encoded, labels=range(num_classes)) # Use all possible labels for consistent matrix size
    print(cm)

    # Plot confusion matrix
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names)
        plt.title('Confusion Matrix on Evaluation Data')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()
    except Exception as plot_e:
        print(f"\nCould not display confusion matrix plot: {plot_e}")

except Exception as e:
    print(f"An error occurred during evaluation: {e}")
    sys.exit()

print("\nEvaluation complete.")
