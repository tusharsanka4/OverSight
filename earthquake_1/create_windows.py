import pandas as pd
import numpy as np
import sys # Import sys

# --- Configuration ---
# Use the LANL sample file now
INPUT_FILE = 'lanl_train_sample.csv'
OUTPUT_FILE = 'labeled_windows.csv' # Output remains the same
WINDOW_SIZE = 40   # Keep window size (adjust if needed based on data sampling rate)
SLIDE_STEP = 5     # Keep slide step
# Threshold for LANL data - needs tuning! Start with a value based on data exploration.
# Let's assume significant activity is above an absolute value of, say, 20 (adjust this!)
ACTIVITY_THRESHOLD = 3

# --- Main Script ---
print(f"Loading data from {INPUT_FILE}...")
try:
    # Read only the acoustic_data column to save memory if file is large
    df = pd.read_csv(INPUT_FILE, usecols=['acoustic_data'])
    # Convert to float32 to save memory and potentially avoid type issues
    df['acoustic_data'] = df['acoustic_data'].astype(np.float32)
except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found.")
    print("Please make sure the lanl_train_sample.csv file is in the same folder.")
    sys.exit() # Use sys.exit()
except ValueError as e:
    print(f"Error reading data: {e}")
    print("Please ensure the 'acoustic_data' column contains only numeric values.")
    sys.exit() # Use sys.exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit() # Use sys.exit()


# Extract only the acoustic data as a NumPy array
data = df['acoustic_data'].values
# We don't have timestamps in this sample, so we'll just use indices
indices = np.arange(len(data))

windows = []
labels = []
start_indices = []

print(f"Creating windows of size {WINDOW_SIZE}...")
# Slide the window across the data
for i in range(0, len(data) - WINDOW_SIZE, SLIDE_STEP):
    window_data = data[i : i + WINDOW_SIZE]
    
    # Check for NaN or infinite values within the window
    if not np.all(np.isfinite(window_data)):
        print(f"Skipping window starting at index {i} due to non-finite values.")
        continue # Skip this window if it contains invalid data

    # Labeling based on activity threshold within the window
    if np.any(np.abs(window_data) > ACTIVITY_THRESHOLD):
        label = 1 # Potential quake signal
    else:
        label = 0 # Background noise

    windows.append(window_data)
    labels.append(label)
    start_indices.append(indices[i]) # Store the starting index instead of timestamp

print(f"Created {len(windows)} labeled windows.")

# --- Save to a new CSV file ---
# Check if any windows were created
if not windows:
    print("No windows were created. Check data length and window size.")
    sys.exit() # Use sys.exit()


# Convert list of windows into a DataFrame
# Use float32 for consistency and memory efficiency
window_df = pd.DataFrame(windows, dtype=np.float32)

# Add descriptive column names
window_df.columns = [f't_{j}' for j in range(WINDOW_SIZE)]

# Add the start index and label
window_df['start_index'] = start_indices
window_df['label'] = labels

# Reorder columns
cols = ['start_index', 'label'] + [col for col in window_df.columns if col not in ['start_index', 'label']]
window_df = window_df[cols]

# Check class balance
label_counts = window_df['label'].value_counts()
print("\nClass balance in generated windows:")
print(label_counts)
if 0 not in label_counts or 1 not in label_counts:
    print("\nWarning: Only one class found. The model might not train effectively.")
    print("Consider adjusting ACTIVITY_THRESHOLD or using a more diverse data sample.")


try:
    window_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully saved labeled data to {OUTPUT_FILE}")
except Exception as e:
    print(f"Error saving file {OUTPUT_FILE}: {e}")
    sys.exit() # Use sys.exit()

