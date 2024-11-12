#!/bin/bash

# Check if the correct number of arguments was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <pdb_path>"
    exit 1
fi

# Activate the MGA conda environment
micromamba activate /home/cch57/.conda/envs/mgltools    

# Define the input PDB path
pdb_path=$1

# Define the output PDBQT path by appending 'qt' to the input path
pdbqt_path="${pdb_path}qt"

# Run prepare_receptor4.py to convert PDB to PDBQT
prepare_receptor4.py -r $pdb_path -o $pdbqt_path -A hydrogens

echo "PDBQT file saved to: $pdbqt_path"
