import kagglehub

# Download latest version
path = kagglehub.dataset_download("s3programmer/flood-risk-in-india")

print("Path to dataset files:", path)