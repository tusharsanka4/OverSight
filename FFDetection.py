import os
import pandas as pd
import numpy as np
import json 
import glob 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model # Used for saving/loading
from joblib import dump # Used for saving the scaler
import kagglehub 

# --- SCRIPT-LEVEL CONSTANT ---
# FIX 1: Define the target column globally so it can be used throughout the script
ACTUAL_TARGET_COLUMN = 'Flood Occurred' 
# -----------------------------

# --- 1. Data Loading and Setup ---

def load_kaggle_data():
    """Downloads the Kaggle dataset and loads the primary CSV file by searching the directory."""
    try:
        print("Downloading dataset from KaggleHub...")
        path_to_files = kagglehub.dataset_download("s3programmer/flood-risk-in-india")
        
        csv_files = glob.glob(os.path.join(path_to_files, '*.csv'))
        
        if not csv_files:
             raise FileNotFoundError(f"Could not locate any CSV files in the downloaded directory: {path_to_files}")
             
        data_file_path = csv_files[0]
        
        if len(csv_files) > 1:
            print(f"Warning: Found multiple CSV files. Loading the first one: {os.path.basename(data_file_path)}")
                 
        print(f"Loading data from: {data_file_path}")
        df = pd.read_csv(data_file_path)
        return df

    except Exception as e:
        print(f"\n--- FATAL ERROR LOADING DATA ---\nDetails: {e}")
        print("Ensure you have run: 'pip install kagglehub' and configured your Kaggle API key.")
        return None

# --- 2. Data Preprocessing (Normalization and Splitting) ---

def preprocess_data(df: pd.DataFrame):
    """Cleans, encodes categorical features, normalizes, and splits the data."""
    
    CATEGORICAL_COLUMNS = ['Land Cover', 'Soil Type']
    
    if ACTUAL_TARGET_COLUMN not in df.columns:
        # This check should now only fail if the data file is wrong.
        raise KeyError(
            f"Target column '{ACTUAL_TARGET_COLUMN}' not found. Cannot proceed with training."
        )

    # 1. Extract Target (y) and Features (X)
    y = df[ACTUAL_TARGET_COLUMN]
    X = df.drop(columns=[ACTUAL_TARGET_COLUMN])

    # --- FEATURE ENGINEERING: ONE-HOT ENCODING ---
    print("\n--- FEATURE ENGINEERING: Encoding Categorical Data ---")
    
    # Use pd.get_dummies to convert text columns into binary columns
    X = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS, drop_first=True)
    
    # --- DATA CLEANUP AND SELECTION ---
    
    # Ensure all columns are numeric after encoding
    X = X.select_dtypes(include=[np.number])
    
    # Handle Missing Values (Imputation): Fill NaNs with the mean of the column
    X = X.fillna(X.mean())

    print(f"Total features after encoding and cleanup: {X.shape[1]}")
    
    # 3. Scale Features (Essential for Neural Networks)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Reshape for LSTM (3D: [Samples, Timesteps, Features])
    TIMESTEPS = 1
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], TIMESTEPS, X_scaled.shape[1])

    # 5. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y.values, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, X_reshaped.shape[2]


# --- 3. Model Definition (LSTM for Binary Classification) ---

def define_model(input_shape):
    """Defines a simple LSTM model for flood classification (0 or 1)."""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy', 
        metrics=['accuracy'] 
    )
    
    return model


# --- 4. Main Execution ---

if __name__ == '__main__':
    # A. Load Data
    data = load_kaggle_data()
    if data is None:
        exit()

    # B. Preprocess Data
    X_train, X_test, y_train, y_test, num_features = preprocess_data(data)
    
    if X_train.shape[0] == 0:
        print("\nInsufficient data after preprocessing. Exiting.")
        exit()
        
    print("\n--- Data Ready ---")
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Feature Dimension: {num_features}")

    # C. Define and Train Model
    model = define_model(input_shape=(X_train.shape[1], num_features))
    
    print("\n--- Training Model ---")
    history = model.fit(
        X_train, y_train,
        epochs=10, 
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # D. Evaluate Model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("\n--------------------------")
    print("Model Training Complete.")
    print(f"Test Loss (Binary Crossentropy): {loss:.4f}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print("--------------------------")

    # E. Save the Model and Scaler for the Agent Tool
    # This block is now outside of any function and can access the necessary 'data' variable.
    
    # 1. Prepare the full feature set for fitting the final scaler
    # Drop the target and re-encode/impute as done in preprocess_data, but on the full dataset
    full_X = data.drop(columns=[ACTUAL_TARGET_COLUMN])
    CATEGORICAL_COLUMNS = ['Land Cover', 'Soil Type']
    full_X = pd.get_dummies(full_X, columns=CATEGORICAL_COLUMNS, drop_first=True)
    full_X = full_X.select_dtypes(include=[np.number])
    full_X = full_X.fillna(full_X.mean())

    # 2. Fit the final scaler
    final_scaler = StandardScaler()
    final_scaler.fit(full_X)
    
    # 3. Save the necessary components
    model.save('flood_model.keras') # Saves the TensorFlow model
    dump(final_scaler, 'flood_scaler.joblib') # Saves the scaler
    
    # 4. Save the feature names (CRITICAL for the Agent Tool)
    feature_names = full_X.columns.tolist()
    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    print("\n✅ Model, Scaler, and Feature Names saved successfully for Agent use.")