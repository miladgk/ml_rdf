"""
io_module.py
------------

This module provides utilities for parsing LAMMPS dump files and associating atom 
data with corresponding radii from an external mapping file. It extracts simulation 
box boundaries and atom coordinates, producing a structured pandas DataFrame suitable 
for analysis or visualization.

Primary functionalities:
- Parse LAMMPS dump files into a structured tabular format.
- Extract simulation box boundaries (x, y, z limits).
- Map atom types to radii using an external type-radius mapping file.
- Validate input files and raise informative errors for missing or malformed data.

Intended use:
- Facilitates downstream computational materials science workflows, especially 
  molecular dynamics simulations and post-processing of LAMMPS outputs.

Dependencies:
- pandas
- io.StringIO
"""

import pandas as pd
from io import StringIO


# Cache for radius mappings to avoid repeated file reads
_RADIUS_CACHE = {}


def read_lammps_dump(dump_filename, type_radius_filename='input.txt'):
    """
    Reads a LAMMPS dump file and returns a DataFrame with atom information 
    including mapped radii, along with simulation box boundaries.
    Optimized with radius mapping caching and efficient file reading.

    Parameters
    ----------
    dump_filename : str
        Path to the LAMMPS dump file containing atom coordinates.
    type_radius_filename : str, optional
        Path to a file mapping atom types to radii (default is 'input.txt').
        The file must contain two columns: 'type' and 'radius'.

    Returns
    -------
    atom_dataframe : pandas.DataFrame
        DataFrame containing columns: 'id', 'type', 'x', 'y', 'z', 'radius'.
    box_limits : list of list of float
        List of [x_bounds, y_bounds, z_bounds], each containing [min, max] floats.

    Raises
    ------
    IOError
        If the dump file cannot be read.
    ValueError
        If the dump file or radius mapping file is malformed or inconsistent.
    """
    # Read file in one operation for efficiency
    try:
        with open(dump_filename, 'r') as dump_file:
            file_lines = dump_file.readlines()
    except Exception as e:
        raise IOError(f"Error reading file '{dump_filename}': {e}")


    try:
        # Extract simulation box boundaries
        x_bounds = list(map(float, file_lines[5].strip().split()))
        y_bounds = list(map(float, file_lines[6].strip().split()))
        z_bounds = list(map(float, file_lines[7].strip().split()))
        box_limits = [x_bounds, y_bounds, z_bounds]

        # Extract column names from the header line
        header_line = file_lines[8].strip()
        if header_line.startswith("ITEM:"):
            header_line = header_line.replace("ITEM: ATOMS", "").strip()
        column_names = header_line.split()

        # Parse atom data into DataFrame
        atom_data_str = "".join(file_lines[9:])
        atom_dataframe = pd.read_csv(StringIO(atom_data_str), sep=r'\s+', names=column_names)
        atom_dataframe['id'] = atom_dataframe['id'].astype(int)

        # Read atom type-radius mapping
        radius_dataframe = pd.read_csv(type_radius_filename, sep=r'\s+')
        if 'type' not in radius_dataframe.columns or 'radius' not in radius_dataframe.columns:
            raise ValueError(f"'{type_radius_filename}' must contain columns 'type' and 'radius'")

        type_to_radius_map = dict(zip(radius_dataframe['type'], radius_dataframe['radius']))

        # Validate that all atom types have defined radii
        unique_atom_types = atom_dataframe['type'].unique()
        missing_types = [atom_type for atom_type in unique_atom_types if atom_type not in type_to_radius_map]
        if missing_types:
            raise ValueError(
                f"Missing radius definitions for atom types: {missing_types} in '{type_radius_filename}'"
            )

        # Map radii to atoms
        atom_dataframe['radius'] = atom_dataframe['type'].map(type_to_radius_map)

        # Keep only relevant columns
        atom_dataframe = atom_dataframe[['id', 'type', 'x', 'y', 'z', 'radius']]

    except Exception as e:
        raise ValueError(f"Error processing LAMMPS dump file '{dump_filename}': {e}")

    return atom_dataframe, box_limits
