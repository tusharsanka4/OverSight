import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import joblib

# --- 0. Configuration ---
DATA_FILE = 'earthquake_alert_data.csv' # ASSUMED FILENAME - Change if needed
FEATURE_COLUMNS = ['magnitude', 'cdi', 'mmi'] # Input features
TARGET_COLUMN = 'alert'                    # Column to predict
MODEL_FILE_NAME = 'alert_classifier_model.keras'
SCALER_FILE_NAME = 'alert_scaler.joblib'
LABEL_ENCODER_FILE_NAME = 'alert_label_encoder.joblib' # To save the label mapping

# --- 1. Load Data ---
print(f"Loading data from {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)
    # Drop rows with missing values in relevant columns
    df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN], inplace=True)
    if df.empty:
        print(f"Error: No valid data found in {DATA_FILE} after dropping missing values.")
        sys.exit()
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found.")
    print("Please make sure the data file exists and the filename is correct.")
    sys.exit()
except Exception as e:
    print(f"Error loading or processing data: {e}")
    sys.exit()

print(f"Loaded {len(df)} rows.")

# Separate features (X) and target (y)
X = df[FEATURE_COLUMNS].astype(np.float32)
y_text = df[TARGET_COLUMN]

# --- 2. Preprocess Data ---
print("\nPreprocessing data...")

# Encode text labels ('green', 'yellow', 'orange', 'red') into integers (0, 1, 2, 3)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)
num_classes = len(label_encoder.classes_)
print(f"Encoded target labels: {dict(zip(label_encoder.classes_, range(num_classes)))}")
print(f"Number of classes: {num_classes}")

# Save the label encoder
print(f"Saving label encoder to {LABEL_ENCODER_FILE_NAME}...")
joblib.dump(label_encoder, LABEL_ENCODER_FILE_NAME)
print("Label encoder saved.")


# Split data before scaling
print("Splitting data into training and testing sets...")
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
except ValueError as e:
    print(f"Error during train_test_split (stratify might require minimum samples per class): {e}")
    print("Trying split without stratify...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Check if split was successful even without stratify
    if len(np.unique(y_train)) < num_classes or len(np.unique(y_test)) < num_classes:
         print("\nWarning: Not all classes might be present in both train and test sets after split.")


# Scale numerical features
print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
print(f"Saving scaler to {SCALER_FILE_NAME}...")
joblib.dump(scaler, SCALER_FILE_NAME)
print("Scaler saved.")

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- 3. Build the DNN Model ---
print("\nBuilding the DNN model...")
model = Sequential([
    Input(shape=(len(FEATURE_COLUMNS),)), # Input shape is number of features
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax') # Output layer: num_classes neurons, softmax for multi-class probabilities
])
model.summary()

# --- 4. Compile the Model ---
print("\nCompiling model...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Use sparse version for integer labels
    metrics=['accuracy']
)

# --- 5. Train the Model ---
print("\nTraining model...")
try:
    history = model.fit(
        X_train,
        y_train,
        epochs=50, # Train for more epochs, adjust as needed
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    print("\nTraining complete.")
except Exception as e:
    print(f"\nAn error occurred during training: {e}")
    sys.exit()

# --- 6. Evaluate the Model ---
print("\n--- Model Evaluation ---")
try:
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    # Get predicted probabilities for each class
    y_pred_probs = model.predict(X_test)
    # Get the class with the highest probability for each sample
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    target_names = label_encoder.classes_
    # Added check for labels present in y_test/y_pred
    unique_test_labels = np.unique(y_test)
    unique_pred_labels = np.unique(y_pred)
    present_labels = np.union1d(unique_test_labels, unique_pred_labels) # All labels seen
    present_target_names = [target_names[i] for i in present_labels]

    # Generate report only for labels that are actually present
    print(classification_report(y_test, y_pred, labels=present_labels, target_names=present_target_names, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, labels=range(num_classes)) # Ensure matrix includes all possible labels
    print(cm)

    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()
    except Exception as plot_e:
        print(f"\nCould not display confusion matrix plot: {plot_e}")

except Exception as e:
    print(f"An error occurred during evaluation: {e}")
    sys.exit()

# --- 7. Save the Fully Functioning Model ---
print(f"\nSaving trained model to {MODEL_FILE_NAME}...")
try:
    model.save(MODEL_FILE_NAME)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")
    sys.exit()