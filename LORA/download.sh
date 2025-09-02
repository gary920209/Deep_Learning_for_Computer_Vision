#!/bin/bash

# URL of the file on Dropbox
DROPBOX_URL="https://www.dropbox.com/scl/fi/b0sn8ndvlks4redvvzrqn/best_model.pt?rlkey=1h1idzonnmfnhh77eus3v3azc&st=77di8spe&dl=1"

# Output filename
OUTPUT_FILE="best_model.pt"

# Download the file
echo "Downloading model from Dropbox..."
curl -L -o $OUTPUT_FILE $DROPBOX_URL

echo "Download complete. File saved as $OUTPUT_FILE."
