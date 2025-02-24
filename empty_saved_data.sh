#!/bin/bash

# Define the target directory
TARGET_DIR="./saved_data"

# Ensure the directory exists
if [ -d "$TARGET_DIR" ]; then
    # Find and delete only files, keeping directories intact
    find "$TARGET_DIR" -type f -delete
    echo "All files in '$TARGET_DIR' and its subdirectories have been removed."
else
    echo "Directory '$TARGET_DIR' does not exist."
fi