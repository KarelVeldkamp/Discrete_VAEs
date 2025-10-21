#!/bin/bash

# Set the path to your parameter file
PARAM_FILE="mixirtpars.txt"

# Loop through each line of the file
while IFS= read -r line; do
  # Skip empty lines
  [ -z "$line" ] && continue

  # Run the Python script with the line's values as arguments
  python3 src/simfit.py $line
done < "$PARAM_FILE"
