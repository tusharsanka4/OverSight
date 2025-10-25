import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys # Ensure sys is imported
import joblib # Import joblib to save the scaler

# --- 0. Configuration ---
LABELED_DATA_FILE = 'labeled_windows.csv'
WINDOW_SIZE = 40  # Must match the window size from create_windows.py
MODEL_FILE_NAME = 'earthquake_ews_model.keras' # Keras' new format
SCALER_FILE_NAME = 'scaler.joblib' # File to save the scaler

# --- 1. Load Data ---
print(f"Loading data from {LABELED_DATA_FILE}...")
try:
    df = pd.read_csv(LABELED_DATA_FILE)
    # Ensure data is float32 for consistency
    numeric_cols = [col for col in df.columns if col.startswith('t_')]
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    df['label'] = df['label'].astype(np.int32)
except FileNotFoundError:
    print(f"Error: {LABELED_DATA_FILE} not found.")
    print("Please make sure you have run create_windows.py successfully.")
    sys.exit()
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit()

# Print some info about the loaded data
print(f"Loaded {len(df)} windows.")
label_counts = df['label'].value_counts()
print("Class balance in loaded data:")
print(label_counts)
if len(label_counts) < 2:
    print("\nWarning: Data contains only one class. Model training may fail or be ineffective.")
    # Decide whether to exit or proceed
    # sys.exit()


# Separate features (the signal data) from labels (0 or 1)
# Drop 'start_index' instead of 'start_time'
try:
    X = df.drop(['start_index', 'label'], axis=1).values
    y = df['label'].values
except KeyError as e:
    print(f"Error: Missing expected column in {LABELED_DATA_FILE}: {e}")
    print("Please ensure create_windows.py generated 'start_index' and 'label' columns.")
    sys.exit()


# --- 2. Preprocess and Split Data ---
print("\nSplitting and scaling data...")

# Check if we have enough data and both classes for splitting
unique_labels_in_y = np.unique(y)
if len(unique_labels_in_y) < 2:
    print("Error: The loaded data contains only one class label.")
    print("Cannot perform stratified split or meaningful training/evaluation.")
    sys.exit() # Exit here as stratified split will fail

if len(df) < 5: # Need enough samples for split
     print("Error: Not enough data samples to split into training and testing sets.")
     sys.exit()

# Split into 80% training and 20% testing
try:
    # Ensure there are at least 2 samples per class for stratify to work reliably
    # Check minimum class count
    min_class_count = label_counts.min()
    required_test_samples = max(1, int(0.2 * min_class_count)) # Need at least 1 sample per class in test

    # Adjust test_size dynamically if a class has very few samples, ensure at least 1 test sample per class
    test_size_adjusted = 0.2
    if min_class_count < 2 / test_size_adjusted: # e.g., if min count < 10 for test_size=0.2
         print(f"\nWarning: Very few samples for class {label_counts.idxmin()} ({min_class_count}). Stratification might be unstable.")
         # Consider alternative handling or proceed with caution


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_adjusted, random_state=42, stratify=y)

    # Verify split results
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        print("\nWarning: Split resulted in only one class in train or test set. Trying non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
             print("Error: Even non-stratified split failed to get both classes in train/test. Check data.")
             sys.exit()


except ValueError as e:
     print(f"Error during train_test_split: {e}")
     print("This might happen if a class has fewer samples than needed for the split.")
     sys.exit()


# Check for non-finite values BEFORE scaling
if not np.all(np.isfinite(X_train)):
    print("Error: Non-finite values (NaN or infinity) found in training features.")
    sys.exit()
if not np.all(np.isfinite(X_test)):
    print("Error: Non-finite values (NaN or infinity) found in test features.")
    sys.exit()


# Scale the data. This is VITAL for neural networks.
try:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # Only transform the test data
    # --- Save the scaler ---
    print(f"Saving scaler to {SCALER_FILE_NAME}...")
    joblib.dump(scaler, SCALER_FILE_NAME)
    print("Scaler saved.")
except ValueError as e:
    print(f"Error during scaling: {e}")
    print("This might happen if a feature column has zero variance (all values are the same).")
    sys.exit()


# Reshape data for the 1D CNN
try:
    X_train = X_train.reshape((X_train.shape[0], WINDOW_SIZE, 1))
    X_test = X_test.reshape((X_test.shape[0], WINDOW_SIZE, 1))
except ValueError as e:
    print(f"Error reshaping data for CNN: {e}")
    print(f"Ensure WINDOW_SIZE ({WINDOW_SIZE}) matches the number of 't_' columns in {LABELED_DATA_FILE}.")
    sys.exit()

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- 3. Build the 1D CNN Model ---
print("\nBuilding the 1D CNN model...")
model = Sequential([
    Input(shape=(WINDOW_SIZE, 1)),
    Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.summary()

# --- 4. Compile the Model ---
print("\nCompiling model...")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- 5. Train the Model ---
print("\nTraining model...")
try:
    # Check if training data has both classes before fitting
    if len(np.unique(y_train)) < 2:
         print("Error: Training data has only one class after split. Cannot train effectively.")
         sys.exit()

    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=16,
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

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # --- ADDED CHECK HERE ---
    unique_labels_test = np.unique(y_test)
    unique_labels_pred = np.unique(y_pred)

    print(f"\nUnique actual labels in test set: {unique_labels_test}")
    print(f"Unique predicted labels in test set: {unique_labels_pred}")

    # Only generate report and matrix if both classes are present in BOTH actual and predicted
    if len(unique_labels_test) == 2 and len(unique_labels_pred) == 2:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Noise (0)', 'Quake (1)'], zero_division=0))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        try:
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Predicted Noise', 'Predicted Quake'],
                        yticklabels=['Actual Noise', 'Actual Quake'])
            plt.title('Confusion Matrix')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            plt.show()
        except Exception as plot_e:
            print(f"\nCould not display confusion matrix plot: {plot_e}")
            print("Ensure matplotlib and seaborn are working correctly.")
    else:
        print("\nSkipping Classification Report and Confusion Matrix.")
        print("Reason: Only one class found in actual test labels or predicted labels.")
        print("This usually means the model only learned to predict one class,")
        print("often due to severe class imbalance or issues in the training data generation.")
        print("Please check the output of create_windows.py and consider adjusting ACTIVITY_THRESHOLD.")
    # --- END ADDED CHECK ---

except Exception as e:
    print(f"An error occurred during evaluation: {e}")
    sys.exit()

# --- 7. Save the Fully Functioning Model ---
print(f"\nSaving trained model to {MODEL_FILE_NAME}...")
try:
    model.save(MODEL_FILE_NAME)
    print("Model saved successfully.")
    print("You can now use this file in another script for live prediction.")
except Exception as e:
    print(f"Error saving model: {e}")
    sys.exit()

