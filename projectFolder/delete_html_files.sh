#!/bin/bash

# Directory containing the .html files
DIR="."

# Check if the directory exists
if [ ! -d "$DIR" ]; then
    echo "Directory $DIR does not exist."
    exit 1
fi

# Loop through all .html files in the directory and delete them
for file in "$DIR"/*.html; do
    # Check if there are any .html files
    if [ -e "$file" ]; then
        rm "$file"
        echo "Deleted $file"
    fi
done

echo "Deletion completed."

