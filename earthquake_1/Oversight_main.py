import pandas as pd
import numpy as np
import sys

# --- Configuration ---
INPUT_QUAKE_FILE = 'stead_earthquake.csv' # Your file with earthquake data
INPUT_NOISE_FILE = 'stead_noise.csv'      # Your file with "false" (noise) data
OUTPUT_FILE = 'labeled_windows.csv'       # The file we will create
WINDOW_SIZE = 40   # How many timesteps to look at
SLIDE_STEP = 5     # How many steps to slide the window

def process_file(filename, label, window_size, slide_step):
    """
    Loads a CSV file, slides a window over its data, and assigns a
    single label (0 or 1) to every window.
    """
    print(f"Processing file: {filename} with label {label}...")
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        print("Please make sure the file is in the same folder.")
        return [], []
    
    # Try to find 'amplitude' column, otherwise use the first column
    if 'amplitude' in df.columns:
        data = df['amplitude'].values
    else:
        # Assume it's a single-column CSV
        data = df.iloc[:, 0].values
        print(f"  > 'amplitude' column not found, using first column.")
        
    windows = []
    labels = []
    
    # Slide the window across the data
    for i in range(0, len(data) - window_size, slide_step):
        window_data = data[i : i + window_size]
        windows.append(window_data)
        labels.append(label)
        
    print(f"  > Created {len(windows)} windows.")
    return windows, labels

# --- Main Script ---
# Process the earthquake file (label=1)
quake_windows, quake_labels = process_file(INPUT_QUAKE_FILE, 1, WINDOW_SIZE, SLIDE_STEP)

# Process the noise file (label=0)
noise_windows, noise_labels = process_file(INPUT_NOISE_FILE, 0, WINDOW_SIZE, SLIDE_STEP)

if not quake_windows and not noise_windows:
    print("No data processed. Exiting.")
    sys.exit()

# Combine the data
all_windows = quake_windows + noise_windows
all_labels = quake_labels + noise_labels

print(f"\nTotal windows created: {len(all_windows)}")
print(f"  Quake windows: {len(quake_windows)}")
print(f"  Noise windows: {len(noise_windows)}")

# --- Save to a new CSV file ---
# Convert our list of windows into a DataFrame
window_df = pd.DataFrame(all_windows)

# Add descriptive column names
window_df.columns = [f't_{j}' for j in range(WINDOW_SIZE)]

# Add the label
window_df['label'] = all_labels

# SHUFFLE the data! This is critical for training.
window_df = window_df.sample(frac=1).reset_index(drop=True)
print("Shuffled all windows and labels.")

# Save the final, combined, labeled, and shuffled dataset
window_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSuccessfully saved labeled data to {OUTPUT_FILE}")

