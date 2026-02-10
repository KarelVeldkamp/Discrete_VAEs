#!/bin/bash

# File: run_simfit.sh
# Description: Run ./src/simfit.py multiple times using arguments from ./pars.txt

# Path to parameter file
PARAM_FILE="./pars.txt"

# Check if pars.txt exists
if [ ! -f "$PARAM_FILE" ]; then
  echo "Error: $PARAM_FILE not found."
  exit 1
fi

# Loop through each line in pars.txt
while IFS= read -r line; do
  # Skip empty lines or lines starting with #
  if [[ -z "$line" || "$line" == \#* ]]; then
    continue
  fi

  echo "Running simfit.py with arguments: $line"

  # Run the Python script with arguments from the line
  python3 ./src/simfit.py $line

  # Check for errors
  if [ $? -ne 0 ]; then
    echo "Error: simfit.py failed for arguments: $line"
  fi

done < "$PARAM_FILE"

