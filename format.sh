#!/bin/bash

# Define the src directory
src_dir="./src"

# Find all Python files in the src directory and save them to a variable
python_files=$(find "$src_dir" -type f -name "*.py")

# Iterate over the list of Python files and format each one with yapf
for file in $python_files; do
    yapf -i "$file"
    echo "Formatted $file"
done

echo "All Python files in src have been formatted."