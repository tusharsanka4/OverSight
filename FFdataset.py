import kagglehub

# Download latest version
path = kagglehub.dataset_download("naiyakhalid/flood-prediction-dataset")

print("Path to dataset files:", path)