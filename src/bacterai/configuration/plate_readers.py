"""
Plate reader data processing for BacterAI.

This module handles reading and processing data from various plate readers
(Biotek, Tecan, etc.) and converting them into the standard mapped_data format
required by BacterAI.
"""

import sys
import os
import re
from typing import Optional, List, Dict, Tuple

import pandas as pd


def read_biotek(filename: str, signal: str = '600') -> pd.DataFrame:
    """
    Read Biotek plate reader Excel file and return melted time-series data.
    
    Args:
        filename: Path to Biotek .xlsx file
        signal: Wavelength signal to extract (default: '600' for OD600)
        
    Returns:
        DataFrame with columns: Time, well, y (OD value), and any temperature columns
    """
    y = pd.read_excel(filename, skiprows=0, parse_dates=False)
    if isinstance(signal, (str, float)):
        signal = int(signal)
    
    # Find where 'Results' section ends
    sp = y.iloc[:, 0] == 'Results'
    sp = sp[sp].index
    y = y.iloc[0:sp[0], :]
    
    # Find sections matching the signal wavelength
    wp = y.iloc[:, 0] == signal
    wp = wp.fillna(False).cumsum()
    z = [group.iloc[2:, 1:] for _, group in y[wp > 0].groupby(wp[wp > 0])]
    
    # Process each section
    for i, df in enumerate(z):
        df.columns = z[i].iloc[0]
        df = df.iloc[1:].copy()
        df.reset_index(inplace=True, drop=True)
        df.loc[:, df.columns != 'Time'] = df.loc[:, df.columns != 'Time'].apply(pd.to_numeric, errors='coerce')
        z[i] = df
    
    # Extract time and temperature columns
    time_and_temp = [col for col in z[0].columns if re.search(r'^T', col, re.IGNORECASE)]
    base_df = z[0]
    other_dfs = [df.drop(time_and_temp, axis=1) for df in z[1:]]
    z_concat = pd.concat([base_df] + other_dfs, axis=1)
    
    # Melt to long format
    z_melted = z_concat.melt(id_vars=time_and_temp, value_name='y', var_name='well')
    z_melted = z_melted.dropna(subset=["Time"])
    z_melted["Time"] = z_melted["Time"].astype("string")
    z_melted["well"] = z_melted["well"].astype("string")
    z_melted["y"] = z_melted["y"].astype("float")
    z_melted["Time"] = pd.to_timedelta(z_melted["Time"]) / pd.Timedelta(days=1)
    
    return z_melted


def read_tecan(filename: str, signal: str = '600') -> pd.DataFrame:
    """
    Read Tecan plate reader .asc file and return well data.
    
    The Tecan format may be a simple two-column text file if only one measurement is taken:
        Raw data    Well positions
        0.0446      A1
        0.0449      B1
        ...
    Or it will have multiple measurement columns (and note that the column names must be defined in the measurement protocol by the user prior):
        Well positions  abs600nm   abs480nm
        A1  0.0446     0.1234
        B1  0.0449     0.0099
        ...
    
    Args:
        filename: Path to Tecan .asc file
        signal: Wavelength signal to extract (default: '600' for OD600). Can accept list of strings.
        
    Returns:
        DataFrame with columns: well, y (OD value)
    """
    
    # Tecan allows footers and headers, so we need to skip non-data lines near the end of files
    num_footer = 0
    footer_lines = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            if re.search(r',|\t', line):
                break
            else:
                num_footer += 1
                footer_lines.append(line)
                
    # Read the file, handling various delimiters (tab, comma) -- not including spaces
    # df = pd.read_csv(filename, sep=r'\s+|,|\t', engine='python', skipfooter=num_footer)
    df = pd.read_csv(filename, sep=r',|\t', engine='python', skipfooter=num_footer)

    # search footer lines for date/time info
    date_time = None
    for line in footer_lines:
        date_match = re.search(r'Date of measurement:\s*(\d{4}-\d{2}-\d{2})', line)
        time_match = re.search(r'Time of measurement:\s*(\d{2}:\d{2}:\d{2})', line)
        if date_match and time_match:
            date_time = f"{date_match.group(1)} {time_match.group(1)}"
            break

    # Expected columns: first is OD values, second is well positions
    if df.shape[1] < 2:
        raise ValueError(f"Tecan file must have at least 2 columns, found {df.shape[1]}")
    
    # search for column "Well positions", other columns represent one or more measurements
    well_col = df.columns == 'Well positions'
    if well_col.sum() == 1:
        well_col = df.columns[well_col][0]
        meas_col = df.columns[df.columns != well_col].to_list()

    # from multiple measurement columns, extract target reading for OD
    if len(meas_col) > 1:
        target_col = None
        for col in meas_col:
            likely_od_terms = '|'.join(signal)
            if re.search(likely_od_terms, col, re.IGNORECASE):
                target_col = col
                break
        if target_col is None:
            target_col = meas_col[0]
    else:
        target_col = meas_col[0]
    
    # Create clean DataFrame
    result = pd.DataFrame({
        'well': df[well_col].astype(str).str.strip(),
        'y': pd.to_numeric(df[target_col], errors='coerce')
    })

    # include additional measure columns
    # NOTE: if needing to modify the delta-OD methods with more info, modify the extract_tecan_delta_od() function
    # NOTE: if needing to measure a different phenotype, create a new extract_tecan_*() function and then modify process_tecan_data() accordingly
    for col in meas_col:
        if col != target_col:
            result[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any rows with missing data
    result = result.dropna()

    # add time column if found
    if date_time is not None:
        result['Time'] = pd.to_datetime(date_time)
    
    return result


def extract_delta_od(biotek_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract delta OD (final - initial) from Biotek time-series data.
    
    Args:
        biotek_df: DataFrame from read_biotek()
        
    Returns:
        DataFrame with columns: well, feature (delta_od)
    """
    initial_od = biotek_df.groupby('well').first()['y']
    final_od = biotek_df.groupby('well').last()['y']
    result = (final_od - initial_od).rename('feature').reset_index()
    return result


def extract_tecan_delta_od(initial_file: str, final_file: str) -> pd.DataFrame:
    """
    Extract delta OD from paired Tecan initial and final readings.
    
    Args:
        initial_file: Path to initial OD reading .asc file
        final_file: Path to final OD reading .asc file
        
    Returns:
        DataFrame with columns: well, feature (delta_od)
    """
    initial_df = read_tecan(initial_file)
    final_df = read_tecan(final_file)
    
    # Merge on well
    merged = pd.merge(initial_df, final_df, on='well', suffixes=('_initial', '_final'))

    # Calculate delta OD
    # NOTE: if modifying the delta OD to include other measures, such as evaporation as estimated from increased dye concentration, do it here
    merged['feature'] = merged['y_final'] - merged['y_initial']
    
    # Return only well and feature columns
    return merged[['well', 'feature']]


def process_tecan_data(
    path: str,
    date: Optional[str],
    round_number: int,
    feature: str = 'delta_od'
) -> pd.DataFrame:
    """
    Process Tecan plate reader data and generate mapped_data CSV.
    
    Tecan produces simpler output than Biotek - typically each time point is saved as
    OD readings in separate .asc files. This function looks for paired files
    (e.g., 'initial_*.asc' and 'final_*.asc' or similar naming patterns).
    Failing that, it will use alphabetical sorting, which will work because each file name should have a simplified date/time stamp.
    
    Args:
        path: Overall experimental path
        date: Date that the experiment request was made (optional)
        round_number: Round for the experiment
        feature: Feature to extract (currently only 'delta_od' supported)
        
    Returns:
        DataFrame ready to save as mapped_data CSV
    """
    if feature != 'delta_od':
        raise NotImplementedError(f"Tecan reader currently only supports 'delta_od', not '{feature}'")
    
    # Determine experiment_request path - try with date subfolder first, then fall back to direct path
    if date:
        date_based_path = os.path.join(path, "experiment_request", date)
        if os.path.exists(date_based_path):
            experiment_request_path = date_based_path
        else:
            # Date provided but folder doesn't exist - fall back to direct path
            experiment_request_path = os.path.join(path, "experiment_request")
    else:
        # No date provided - check either RoundN/experiment_request/ or experiment_request/RoundN/
        experiment_request_path = os.path.join(path, "experiment_request")
        if os.path.exists(experiment_request_path):
            experiment_request_path = os.path.join(experiment_request_path, f"Round{round_number}")
        else:
            experiment_request_path = os.path.join(path, f"Round{round_number}", "experiment_request")
    
    plate_maps_path = os.path.join(experiment_request_path, "plate_maps")
    # round_folder = os.path.join(path, f"Round{round_number}")

    # Load plate maps and file ID mappings
    instructions = [d for d in os.listdir(plate_maps_path) if os.path.isdir(os.path.join(plate_maps_path, d))]
    
    map_files = [pd.read_csv(os.path.join(plate_maps_path, instruction, "map.csv")) for instruction in instructions]
    maps_combined = pd.concat(map_files)

    plate_to_file_id_files = [pd.read_csv(os.path.join(plate_maps_path, instruction, f"{instruction}_plate_to_file_id.csv")) 
                               for instruction in instructions]
    plate_to_file_id_combined = pd.concat(plate_to_file_id_files)

    # Get unique file IDs from the plate_to_file_id mappings
    unique_file_ids = plate_to_file_id_combined['file_id'].unique().tolist()
    
    data_path = os.path.join(experiment_request_path, "data")

    # Process each file ID
    final_dfs = []
    for fid in unique_file_ids:
        # # Load exception file if exists
        # exception_file = [f for f in os.listdir(data_path) if fid in f and "exception" in f]
        
        # exceptions = []
        # if len(exception_file) >= 1:
        #     for eid in exception_file:
        #         exception = pd.read_csv(os.path.join(data_path, eid))
        #         exceptions.append(exception)
        #     exceptions = pd.concat(exceptions, ignore_index=True)
        # else:
        #     exceptions = None
        exceptions = None   # there should be no exceptions for Tecan data, errors usually halt a run
       
        # Find Tecan .asc files - look for initial and final pairs
        tecan_files = [f for f in os.listdir(data_path) if fid in f and f.endswith('.asc')]
        
        all_plate_bad = False

        if len(tecan_files) == 0:
            # No file - create empty DF with all wells marked as bad (so will be flagged for redo)
            print(f"Warning: No .asc files found for file ID: {fid}. All wells will be marked for redo", file=sys.stderr)
            all_plate_bad = True
            final_df = pd.DataFrame({
                'well': [f"{row}{col:d}" for row in 'ABCDEFGH' for col in range(1,13)],
                'feature': [0.0]*96
            })
            # raise ValueError(f"No .asc files found for file ID: {fid}")
        elif len(tecan_files) == 1:
            # Single file - treat as final OD only (no delta calculation possible)
            # Just use raw values as "feature"
            tecan_df = read_tecan(os.path.join(data_path, tecan_files[0]))
            final_df = tecan_df.rename(columns={'y': 'feature'})
        elif len(tecan_files) == 2:
            # Two files - assume initial and final
            # Try to identify which is which based on filename
            initial_file = None
            final_file = None
            
            for f in tecan_files:
                f_lower = f.lower()
                if 'initial' in f_lower or 'init' in f_lower or 'start' in f_lower or 't0' in f_lower:
                    initial_file = os.path.join(data_path, f)
                elif 'final' in f_lower or 'end' in f_lower or 't1' in f_lower or 'tf' in f_lower:
                    final_file = os.path.join(data_path, f)
            
            # If we couldn't identify by name, use alphabetical order
            if initial_file is None or final_file is None:
                sorted_files = sorted(tecan_files)
                initial_file = os.path.join(data_path, sorted_files[0])
                final_file = os.path.join(data_path, sorted_files[1])
            
            if feature == 'delta_od':
                final_df = extract_tecan_delta_od(initial_file, final_file)
        else:
            # More than two files - use alphabetical order and pick first and last
            # This assumes that files are named with date/time stamps, a reasonable assumption
            sorted_files = sorted(tecan_files)
            initial_file = os.path.join(data_path, sorted_files[0])
            final_file = os.path.join(data_path, sorted_files[-1])

            if feature == 'delta_od':
                final_df = extract_tecan_delta_od(initial_file, final_file)
        
        # Add metadata
        final_df['file_id'] = fid
        if exceptions is not None:
            final_df['bad'] = final_df['well'].isin(exceptions['Destination Well']).astype(int)
        elif all_plate_bad == True:
            final_df['bad'] = 1
        else:
            final_df['bad'] = 0
        
        final_dfs.append(final_df)        
   
    # Combine all results
    final_dfs = pd.concat(final_dfs, ignore_index=True)
    
    # Merge with plate maps
    result = pd.merge(final_dfs, plate_to_file_id_combined, on='file_id', how='left')

    # if column "environment" exists, remove it to avoid duplication issues in final mapped_data
    if 'environment' in maps_combined.columns:
        maps_combined = maps_combined.drop(columns=['environment'])

    result = pd.merge(result, maps_combined, left_on=['parent_plate', 'well'], 
                     right_on=['parent_plate', 'parent_well'], how='left')
    
    result = result.dropna(subset=["solution_id"])

    # Extract experiment number and fill controls with 9999
    result['experiment_number'] = result['solution_id'].str.extract(r'expt(\d+)').fillna(value=9999).astype(int)

    # negative delta OD values should be 0
    if feature == 'delta_od':
        result.loc[(result['feature'] < 0) & (result['plate_blank'] == False), 'feature'] = 0

    # Keep only required columns
    names_to_keep = ['feature', 'bad', 'plate_control', 'plate_blank', 'parent_plate', 
                     'experiment_number', 'strain', 'environment']
    result = result[names_to_keep]

    return result


def process_biotek_data(
    path: str,
    date: Optional[str],
    round_number: int,
    signal: int,
    feature: str
) -> pd.DataFrame:
    """
    Process Biotek plate reader data and generate mapped_data CSV.
    
    Args:
        path: Overall experimental path
        date: Date that the experiment request was made (optional)
        round_number: Round for the experiment
        signal: Wavelength of plate reader measurements
        feature: Feature to extract ('delta_od', 'growth_rate', 'lag_time')
        
    Returns:
        DataFrame ready to save as mapped_data CSV
    """
    # Determine experiment_request path - try with date subfolder first, then fall back to direct path
    if date:
        date_based_path = os.path.join(path, "experiment_request", date)
        if os.path.exists(date_based_path):
            experiment_request_path = date_based_path
        else:
            # Date provided but folder doesn't exist - fall back to direct path
            experiment_request_path = os.path.join(path, "experiment_request")
    else:
        # No date provided - check either experiment_request/RoundN/ or RoundN/experiment_request/
        experiment_request_path = os.path.join(path, "experiment_request")
        if os.path.exists(experiment_request_path):
            experiment_request_path = os.path.join(experiment_request_path, f"Round{round_number}")
        else:
            experiment_request_path = os.path.join(path, f"Round{round_number}", "experiment_request")
    
    plate_maps_path = os.path.join(experiment_request_path, "plate_maps")
    round_folder = os.path.join(path, f"Round{round_number}")

    # Load plate maps and file ID mappings
    instructions = [d for d in os.listdir(plate_maps_path) if os.path.isdir(os.path.join(plate_maps_path, d))]
    
    map_files = [pd.read_csv(os.path.join(plate_maps_path, instruction, "map.csv")) for instruction in instructions]
    maps_combined = pd.concat(map_files)

    plate_to_file_id_files = [pd.read_csv(os.path.join(plate_maps_path, instruction, "plate_to_file_id.csv")) 
                               for instruction in instructions]
    plate_to_file_id_combined = pd.concat(plate_to_file_id_files)

    # Get unique file IDs from worklists
    worklists_path = os.path.join(experiment_request_path, "worklists")
    unique_file_ids = []
    for instruction in instructions:
        csv_files = [f for f in os.listdir(os.path.join(worklists_path, instruction)) if f.endswith('.csv')]
        unique_ids = {re.split(r'[_.,]', f)[0] for f in csv_files}
        unique_file_ids += unique_ids
    
    data_path = os.path.join(experiment_request_path, "data")

    # Process each file ID
    final_dfs = []
    for fid in unique_file_ids:
        # Load exception file if exists
        exception_file = [f for f in os.listdir(data_path) if fid in f and "exception" in f]
        
        exceptions = []
        if len(exception_file) >= 1:
            for eid in exception_file:
                exception = pd.read_csv(os.path.join(data_path, eid))
                exceptions.append(exception)
            exceptions = pd.concat(exceptions, ignore_index=True)
        else:
            exceptions = None
       
        # Read Biotek file
        biotek_file = [f for f in os.listdir(data_path) if fid in f and f.endswith('.xlsx')]
        if len(biotek_file) == 1:
            biotek_df = read_biotek(os.path.join(data_path, biotek_file[0]), str(signal))
        else:
            raise ValueError(f"Number of possible Excel files containing {fid} is either none or more than one")

        # Extract feature
        if feature == 'delta_od':
            final_df = extract_delta_od(biotek_df)
        elif feature == 'growth_rate':
            # Placeholder: Calculate growth_rate
            raise NotImplementedError("growth_rate feature not yet implemented")
        elif feature == 'lag_time':
            # Placeholder: Calculate lag_time
            raise NotImplementedError("lag_time feature not yet implemented")
        else:
            raise ValueError(f"Unknown feature: {feature}")
        
        if final_df is None:
            raise ValueError(f"No feature measured for {fid}")
        
        # Add metadata
        final_df['file_id'] = fid
        if exceptions is not None:
            final_df['bad'] = final_df['well'].isin(exceptions['Destination Well']).astype(int)
        else:
            final_df['bad'] = 0
        
        final_dfs.append(final_df)        
   
    # Combine all results
    final_dfs = pd.concat(final_dfs, ignore_index=True)
    
    # Merge with plate maps
    result = pd.merge(final_dfs, plate_to_file_id_combined, on='file_id', how='left')
    result = pd.merge(result, maps_combined, left_on=['parent_plate', 'well'], 
                     right_on=['parent_plate', 'parent_well'], how='left')
    
    result = result.dropna(subset=["solution_id"])

    # Extract experiment number and fill controls with 9999
    result['experiment_number'] = result['solution_id'].str.extract(r'expt(\d+)').fillna(value=9999).astype(int)

    # negative delta OD values should be 0
    if feature == 'delta_od':
        result.loc[(result['feature'] < 0) & (result['plate_blank'] == False), 'feature'] = 0
    
    # Keep only required columns
    names_to_keep = ['feature', 'bad', 'plate_control', 'plate_blank', 'parent_plate', 
                     'experiment_number', 'strain', 'environment']
    result = result[names_to_keep]

    return result


def save_mapped_data(
    result_df: pd.DataFrame,
    path: str,
    date: str,
    round_number: int,
    reader_type: str,
    feature: str
) -> str:
    """
    Save processed data as mapped_data CSV file.
    
    Args:
        result_df: Processed DataFrame
        path: Overall experimental path
        date: Date of experiment request
        round_number: Round number
        reader_type: Type of plate reader ('biotek', 'tecan')
        feature: Extracted feature name
        
    Returns:
        Path to saved file
    """
    round_folder = os.path.join(path, f"Round{round_number}")
    out_file = f'mapped_data_{date}_{reader_type}_{feature}_data.csv'
    out_path = os.path.join(round_folder, out_file)
    
    result_df.to_csv(out_path, index=False)
    
    return out_path
