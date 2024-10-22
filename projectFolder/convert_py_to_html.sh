#!/bin/bash

# Check if pygmentize is installed
if ! command -v pygmentize &> /dev/null; then
    echo "pygmentize could not be found. Please install Pygments."
    exit 1
fi

# Directory containing the .py files
DIR="."

# Loop through all .py files in the directory
for file in "$DIR"/*.py; do
    # Check if there are any .py files
    if [ ! -e "$file" ]; then
        echo "No .py files found in the directory."
        exit 0
    fi

    # Get the base name of the file without the extension
    base_name=$(basename "$file" .py)
    
    # Convert the .py file to .html
    pygmentize -f html -O full -o "$DIR/$base_name.html" "$file"
    
    echo "Converted $file to $base_name.html"
done

echo "Conversion completed."

