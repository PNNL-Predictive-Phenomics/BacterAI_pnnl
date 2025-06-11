import pandas as pd
import os
import re
import argparse

def read_biotek(filename, signal='600'):
    y = pd.read_excel(filename, skiprows=0, parse_dates = False)
    if isinstance(signal, (str, float)):
        signal = int(signal)
    sp = y.iloc[:, 0] == 'Results'
    sp = sp[sp].index
    y = y.iloc[0:sp[0],:]
    wp = y.iloc[:, 0] == signal
    wp = wp.fillna(False).cumsum()
    z = [group.iloc[2:, 1:] for _, group in y[wp > 0].groupby(wp[wp > 0])]
    for i, df in enumerate(z):
        df.columns = z[i].iloc[0]
        df = df.iloc[1:]
        df.reset_index(inplace = True, drop = True)
        df.loc[:, df.columns != 'Time'] = df.loc[:, df.columns != 'Time'].apply(pd.to_numeric, errors='coerce')
        z[i] = df
    time_and_temp = [col for col in z[0].columns if re.search(r'^T', col, re.IGNORECASE)]
    base_df = z[0]
    other_dfs = [df.drop(time_and_temp, axis=1) for df in z[1:]]
    z_concat = pd.concat([base_df] + other_dfs, axis=1)
    z_melted = z_concat.melt(id_vars=time_and_temp, value_name='y', var_name='well')
    z_melted = z_melted.dropna(subset = ["Time"])
    z_melted["Time"] = z_melted["Time"].astype("string")
    z_melted["well"] = z_melted["well"].astype("string")
    z_melted["y"] = z_melted["y"].astype("float")
    z_melted["Time"] = pd.to_timedelta(z_melted["Time"]) / pd.Timedelta(days = 1)
    return z_melted


def main(path, date, round_number, signal, feature):
    experiment_request_path = os.path.join(path, "experiment_request", date)
    plate_maps_path = os.path.join(experiment_request_path, "plate_maps")
    
    round_folder = os.path.join(path, f"Round{round_number}")

    instructions = [d for d in os.listdir(plate_maps_path) if os.path.isdir(os.path.join(plate_maps_path, d))]
    
    map_files = [pd.read_csv(os.path.join(plate_maps_path, instruction, "map.csv")) for instruction in instructions]
    maps_combined = pd.concat(map_files)

    plate_to_file_id_files = [pd.read_csv(os.path.join(plate_maps_path, instruction, "plate_to_file_id.csv")) for instruction in instructions]
    plate_to_file_id_combined = pd.concat(plate_to_file_id_files)

    worklists_path = os.path.join(experiment_request_path, "worklists")
    unique_file_ids = []
    for instruction in instructions:
        csv_files = [f for f in os.listdir(os.path.join(worklists_path, instruction)) if f.endswith('.csv')]
        unique_ids = {re.split(r'[_.,]', f)[0] for f in csv_files}
        unique_file_ids += unique_ids
    
    data_path =  os.path.join(experiment_request_path, "data")

    final_dfs = []
    for fid in unique_file_ids:

        exception_file = [f for f in os.listdir(data_path) if fid in f and "exception" in f][0]
        exceptions = pd.read_csv(os.path.join(data_path, exception_file)) if exception_file else None
       
        biotek_file = [f for f in os.listdir(data_path) if fid in f and f.endswith('.xlsx')]
        if len(biotek_file) == 1:
            biotek_df = read_biotek(os.path.join(data_path, biotek_file[0]), signal)
        else:
            raise ValueError(f"Number of possible Excel files containing {fid} is either none or more than one")

        if feature == 'delta_od':
            initial_od = biotek_df.groupby('well').first()['y']
            final_od = biotek_df.groupby('well').last()['y']
            final_df = (final_od - initial_od).rename('feature').reset_index()
        elif feature == 'growth_rate':
            # Placeholder: Calculate growth_rate
            pass
        elif feature == 'lag_time':
            # Placeholder: Calculate lag_time
            pass
        
        if final_df is None:
            raise ValueError(f"No feature measured for {fid}")
        else:
            final_df['file_id'] = fid
            if exceptions is not None:
                final_df['bad'] = final_df['well'].isin(exceptions['Destination Well']).astype(int)
            else:
                final_df['bad'] = 0
            
            final_dfs.append(final_df)        
       
    final_dfs = pd.concat(final_dfs, ignore_index = True)
    
    result = pd.merge(final_dfs, plate_to_file_id_combined, on='file_id', how='left')
    result = pd.merge(result, maps_combined, left_on=['parent_plate', 'well'], right_on=['parent_plate', 'parent_well'], how='left')
    
    result = result.dropna(subset=["solution_id"])

    # extract experiment number and fill controls with 9999 so they won't match to experiment request df
    result['experiment_number'] = result['solution_id'].str.extract(r'expt(\d+)').fillna(value = 10000).astype(int) - 1
    
    names_to_keep = ['feature', 'bad', 'plate_control', 'plate_blank', 'parent_plate', 'experiment_number', 'strain', 'environment']
    result = result[names_to_keep]

    out_file = 'mapped_data_' + date + '_biotek_' + feature + '_data.csv'
    result.to_csv(os.path.join(round_folder, out_file), index = False)

    print(f"Successfully wrote Biotek output:\r\n  Location: {round_folder}\r\n  File: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Biotek data and extract a growth curve feature')
    parser.add_argument('path', type=str, help='Overall experimental path')
    parser.add_argument('-d', '--date', required=True, type=str, help='Date that the experiment request was made')
    parser.add_argument('-r', '--round_number', required=True, type=int, help='Round for the experiment')
    parser.add_argument('-s', '--signal', required=True, type=int, help='Wavelength of plate reader measurements')
    parser.add_argument('-f', '--feature', required=True, type=str, choices=['delta_od', 'growth_rate', 'lag_time'], help='Feature to extract')
    
    args = parser.parse_args()
    main(args.path, args.date, args.round_number, args.signal, args.feature)
    
    # example run:
    # python biotek_feature_extract.py /path/to/experiment/ -d "2025-04-01"" -r 1 -s 600 -f "delta_od"
