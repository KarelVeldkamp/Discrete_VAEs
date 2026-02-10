#!/bin/bash

# File: run_LCA.sh
# Description: Run ./Rscripts/LCA.R multiple times using arguments from pars.txt

PARAM_FILE="./gdinapars.txt"

# Check if pars.txt exists
if [ ! -f "$PARAM_FILE" ]; then
  echo "Error: $PARAM_FILE not found."
  exit 1
fi

# Handle Ctrl+C (SIGINT)
trap "echo -e '\n[!] Ctrl+C detected — stopping script.'; exit 130" INT

# Loop through each line in pars.txt
while IFS= read -r line; do
  # Skip empty lines or lines starting with #
  if [[ -z "$line" || "$line" == \#* ]]; then
    continue
  fi

  echo "Running GDINA.R with arguments: $line"

  # Run the R script with arguments from the line
  Rscript ./Rscripts/GDINA.R $line
  status=$?

  if [ $status -ne 0 ]; then
    echo "Error: Rscript failed for arguments: $line"
  fi

done < "$PARAM_FILE"

