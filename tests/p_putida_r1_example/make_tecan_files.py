#!/usr/bin/env python
"""
PlatePlan experiment scheduler for Tecan liquid handler.

This script generates Tecan worklists and experiment layouts for growth boundary experiments.
Run from command line or import functions for custom workflows.

Usage:
    python run_plateplan_refactored.py --round 1 --total-volume "274 ul" \\
        --inoculation-volume "2 ul" --stock-x 4 --prefix "PputidaExp"
"""

import os
import datetime
import copy
import argparse
import shutil
import pandas as pd

from plateplan import (constants, ingredients, units, scheduling_tecan, makeids, mapper, tecan)


def load_experiment_data(round_number, base_path=None):
    """Load experiment metadata from BacterAI batch file.
    
    Parameters
    ----------
    round_number : int
        Experiment round number (e.g., 1, 2, 3)
    base_path : str, optional
        Base directory path. If None, uses current working directory.
        
    Returns
    -------
    pd.DataFrame
        Batch metadata with experiment conditions
    """
    if base_path is None:
        base_path = os.getcwd()
        
    meta_path = os.path.join(base_path, f"Round{round_number}")
    meta_file = list(filter(lambda x: 'batch_meta' in x, os.listdir(meta_path)))[0]
    meta_file_path = os.path.join(meta_path, meta_file)
    batch_pd = pd.read_csv(meta_file_path, index_col=None)

    # Clean up unnecessary columns
    batch_pd.drop(["type", "direction", "frontier_type", "growth_pred", "var", "is_redo", "round"], 
                  axis=1, inplace=True)
    batch_pd.drop("experiment_number", axis=1, inplace=True, errors="ignore")
    
    return batch_pd


def load_ingredients(base_path=None):
    """Load ingredient definitions from CSV file.
    
    Parameters
    ----------
    base_path : str, optional
        Base directory path. If None, uses current working directory.
        
    Returns
    -------
    pd.DataFrame
        Filtered ingredient data (excludes strains, media, and pH)
    """
    if base_path is None:
        base_path = os.getcwd()
        
    ingredients_full_path = os.path.join(base_path, 'ingredients.csv')
    ingredients_pd = pd.read_csv(ingredients_full_path)

    # Filter out non-stock ingredients
    ingredients_pd = ingredients_pd[ingredients_pd["TYPE"] != "strain"]
    ingredients_pd = ingredients_pd[ingredients_pd["TYPE"] != "media"]
    ingredients_pd = ingredients_pd[ingredients_pd["REAGENT"] != "pH"]
    
    return ingredients_pd


def create_reagents(ingredients_pd):
    """Create reagent objects from ingredients dataframe.
    
    Parameters
    ----------
    ingredients_pd : pd.DataFrame
        Ingredient data with REAGENT and MW columns
        
    Returns
    -------
    dict
        Dictionary mapping reagent IDs to Reagent objects
    """
    reagents = dict()
    for rid in ingredients_pd["REAGENT"].dropna().unique():
        mw = ingredients_pd.loc[ingredients_pd["REAGENT"] == rid, "MW"].iloc[0]
        mw = str(mw) + " g/mol"
        reagents[rid] = ingredients.Reagent(id_=rid, name=rid, molecular_weight=mw)
    
    return reagents


def create_stocks(ingredients_pd, stock_x_range=None):
    """Create stock solutions from ingredient data.
    
    Parameters
    ----------
    ingredients_pd : pd.DataFrame
        Ingredient data with CONDITION, REAGENT, STOCK_X, and MAX_VALUE columns
    stock_x_range : int, optional
        Default stock concentration multiplier if not specified per reagent
        
    Returns
    -------
    dict
        Dictionary mapping stock IDs to Stock objects
    """
    stock_pd = ingredients_pd[ingredients_pd["TYPE"].isin(["binary", "quantitative", "semi-quantitative"])]
    stocks = {}
    
    # Specify several stock values for reagents for large range of possible concentrations
    stxr = [2] if stock_x_range is None else stock_x_range
    
    for r in stock_pd["CONDITION"].unique():
        # Get all reagents and concentrations for this condition
        cond_data = stock_pd[stock_pd["CONDITION"] == r]
        stx = cond_data[cond_data["REAGENT"].isna()]["STOCK_X"].iloc[0] if not cond_data[cond_data["REAGENT"].isna()]["STOCK_X"].empty else None
        
        # Build the reagents dict for this condition
        rgt = {}
        for _, row in cond_data.iterrows():
            if pd.isna(row["REAGENT"]):
                continue 
            reagent_id = row["REAGENT"]
            conc_value = row["MAX_VALUE"]
            conc_str = str(units.parse(conc_value, 'mmol/l'))
            rgt[reagent_id] = conc_str
        
        # Create solution with all reagents for this condition, then create stock(s)
        # NOTE: that multiple stock ranges are only used for reagents without specified STOCK_X value
        id_ = r  # Use condition name as the stock ID
        stock_x_i = [stx] if stx is not None else stxr
    
        for X in stock_x_i:
            soln_i = ingredients.Solution(reagents=rgt, solvent="water", id_=id_)
            si = ingredients.Stock(
                ingredient=soln_i.concentrated(X), 
                date_made=datetime.date.today(),
                date_expires=datetime.date.today() + datetime.timedelta(1 * 365 / 12),
                labware=constants.labware['Conical50ml'],
                quantity=units.parse('10 mL'),
                location='TBD_on_Tecan',
            )
            stocks[si.ingredient.id] = si
    
    return stocks


def create_solutions(batch_pd):
    """Create solution objects from batch experiment data.
    
    Parameters
    ----------
    batch_pd : pd.DataFrame
        Batch metadata with reagent concentrations
        
    Returns
    -------
    list
        List of Solution objects for each experiment
    """
    solutions = []

    for i in batch_pd.index:
        rgt = batch_pd.iloc[i]
        rgt = pd.DataFrame({"REAGENT": rgt.index.tolist(), "CONC": rgt.tolist()})
        rgt = rgt[rgt["CONC"] > 0]

        if rgt["REAGENT"].duplicated().any():
            rgt = rgt.groupby(["REAGENT"], as_index=False)["CONC"].sum()

        rgt["SOLN_CONC"] = list(rgt["CONC"].astype(str) + " mmol/l")
        rgt = rgt.set_index("REAGENT")["SOLN_CONC"].to_dict()
        id_ = "expt" + str(i + 1)

        soln = ingredients.Solution(reagents=rgt, id_=id_)
        solutions.append(soln)

    # For low-O2 conditions, include necessary ingredients
    for s in solutions:
        if "o2" not in s.reagents:
            s.reagents["oxyrase"] = units.parse("1 mmol/l")
            s.reagents["glycerophosphoric_acid"] = units.parse("10 mmol/l")
            s.reagents["sodium_formate"] = units.parse("10 mmol/l")
        else:
            del s.reagents["o2"]

    # Add pH values to the name of each experiment so that they can be grouped
    for s in solutions:
        ph_val = str(s.reagents["pH"].value).replace('.0', '')
        s.id = s.id + "_pH::" + ph_val
        del s.reagents["pH"]
    
    return solutions


def main(round_number, total_volume, inoculation_volume, media_x, experiment_prefix, base_path=None,
         strains = None, environments = None, stock_x_range=None, stock_containers="Conical50ml"):
    """Main execution function for experiment scheduling.
    
    Parameters
    ----------
    round_number : int
        Experiment round number
    total_volume : str
        Total volume per well (e.g., "274 ul")
    inoculation_volume : str
        Inoculation volume per well (e.g., "2 ul")
    media_x : int
        Stock concentration multiplier FOR BASELINE MEDIA (e.g., 4 for 4X stock)
    experiment_prefix : str
        Short prefix for experiment identification
    base_path : str, optional
        Base directory path. If None, uses current working directory.
    strains : dict
        Dictionary mapping strain short names to full names
    environments : dict
        Dictionary mapping environment short names to full names
    stock_x_range : list, optional
        List of stock concentration multipliers to use for creating stocks.
        If None, defaults to 2X.
    stock_containers : str, optional
        Labware type for stock containers (from constants.labware).
        Default is "Conical50ml".
    """
    if base_path is None:
        base_path = os.getcwd()
    
    # Parse volumes
    total_vol = units.parse(total_volume)
    inoc_vol = units.parse(inoculation_volume)

    tv = total_vol.base_value
    iv = inoc_vol.base_value

    tvu = total_vol.prefix + total_vol.unit
    
    # Calculate working volume: total - inoculation - (total / media_X)
    wv = tv - iv - (tv / media_x)
    working_vol = units.parse(f"{wv} l").convert(tvu)

    print(f"\n{'='*60}")
    print(f"PlatePlan Experiment Scheduler - Round {round_number}")
    print(f"{'='*60}")
    print(f"Total volume:        {total_vol}")
    print(f"Inoculation volume:  {inoc_vol}")
    print(f"Media concentration: {media_x}X")
    print(f"Working volume:      {working_vol}")
    print(f"Stock conc. range:   {stock_x_range}")
    print(f"Experiment prefix:   {experiment_prefix}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading experiment data...")
    batch_pd = load_experiment_data(round_number, base_path)
    ingredients_pd = load_ingredients(base_path)
    
    # Create reagents, stocks, and solutions
    print("Creating reagents...")
    reagents = create_reagents(ingredients_pd)
    
    print("Creating stocks...")
    stocks = create_stocks(ingredients_pd, stock_x_range=stock_x_range)
    
    print("Creating solutions...")
    solutions = create_solutions(batch_pd)
    
    # Add after-the-fact controls and special stocks
    
    # Add orange-G reagent and stock for evaporation control
    reagents["orange_g"] = ingredients.Reagent(
        id_="orange_g", 
        name="orange_g", 
        molecular_weight="452.36 g/mol"
    )
    
    og_soln = ingredients.Solution(
        reagents={"orange_g": "2.2 mmol/l"},
        solvent="water",
        id_="orange_g",
    )
    
    # make up a 50X stock solution of orange-G, which will require a small volume pipetting dispense for target concentration
    # For now, I want to keep orange-G out of the sewer waste stream
    orange_g_stock = ingredients.Stock(
        ingredient=og_soln.concentrated(50),  # working conc = 0.001%, stock = 0.05%
        date_made=datetime.date.today(),
        date_expires=datetime.date.today() + datetime.timedelta(6 * 365 / 12),
        quantity=units.parse("50 ml"),
        labware="Conical50ml",
    )

    stocks[orange_g_stock.ingredient.id] = orange_g_stock

    # Add placeholder reagent / 2X stock for plate blank
    reagents["placeholder"] = ingredients.Reagent(
        id_="placeholder", 
        name="placeholder", 
        molecular_weight="100 g/mol"
    )
    
    placeholder_soln = ingredients.Solution(
        reagents={"placeholder": "2.2 mmol/l"}, # any concentration is fine here
        solvent="water",
        id_="placeholder",
    )
    
    placeholder_stock = ingredients.Stock(
        ingredient=placeholder_soln.concentrated(2),  
        date_made=datetime.date.today(),
        date_expires=datetime.date.today() + datetime.timedelta(6 * 365 / 12),
        quantity=units.parse("50 ml"),
        labware="Conical50ml",
    )
    
    stocks[placeholder_stock.ingredient.id] = placeholder_stock
    
    # Create control solutions
    pc = ingredients.Solution(
        reagents={"d_glucose": "4 mmol/L", "ammonium_chloride": "1 mmol/L"}, 
        id_="plate_control"
    )
    
    pb = ingredients.Solution(
        reagents={'placeholder': '2.2 mmol/l'},   # NOTE: "placeholder" and will be removed later
        id_="plate_blank"
    )
    
    ec = ingredients.Solution(
        reagents={'orange_g': '2.2 mmol/l'}, 
        id_="evap_control"
    )
    
    # Define O2 control (plate control + O2 limitation)
    oc = copy.deepcopy(pc)
    oc.id = "o2_control"
    oc.reagents["oxyrase"] = units.parse("1 mmol/l")
    oc.reagents["glycerophosphoric_acid"] = units.parse("10 mmol/l")
    oc.reagents["sodium_formate"] = units.parse("10 mmol/l")
    
    # Add all controls to solution list
    [solutions.append(s) for s in [pc, pb, oc, ec]]
    
    # Schedule the experiment
    print("\nScheduling Tecan liquid handler operations...")
    print(f"  - Solutions (w/ ctrls): {len(solutions)}")
    print(f"  - Stocks: {len(stocks)}")

    fill_excess = "water"
    
    plates, instructions, layout = scheduling_tecan.schedule_tecan(
        solutions,
        ['PpJE3959'],
        environments, 
        stocks,
        excess=fill_excess,
        plate=constants.labware["WP96_325ul_no_corner"],
        replicates=3,
        plate_control=[
            ("plate_control", 3),
            ("o2_control", 3),
            ("evap_control", 3)
        ],
        plate_blank=("plate_blank", 3),
        randomize=True,
        plate_prefix=f"{experiment_prefix}_",
        total_volume=str(total_vol), 
        working_volume=str(working_vol),
        quantity_units="mmol/L",
        min_volume="10 ul",
        max_stocks=16,
        minimize_stocks=False, 
        quiet=True,
    )

    print(f"  - Plates: {len(instructions[0].plates)}")


    # Organize and save output
    exp_id = makeids.unique_id(prefix=experiment_prefix)
    instruction_set = scheduling_tecan.InstructionSet(
        id_=exp_id, 
        instructions=instructions, 
        owner="Stone_PPI",
    )
    
    dtt = datetime.date.today().strftime("%Y-%m-%d")
    output_filepath = os.path.join(base_path, f"Round{round_number}", "experiment_request")
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    # Tecan worktable labware positions
    wt = constants.tecan_worktable

    # export location for all labware needed -- will add to file throughout this process
    labware_path = os.path.join(
        output_filepath, "plate_maps", instruction_set.id, 
        f"{instruction_set.id}_labware_list.txt"
    )
    os.makedirs(os.path.dirname(labware_path), exist_ok=True)
    
    # Add experiment plates to labware list
    for plate, file in instructions[0].plates:
        with open(labware_path, 'a') as f:
            line = f"{file.file_id}\t96 Well Flat_BacterAI\t{wt['plate']['name']}\t{wt['plate']['current']}\n"
            f.write(line)
        wt['plate']['current'] += 1

    # for plate blanks, replace the placeholder with water in the worklist
    for plate, file in instructions[0].plates:
        wl = file.worklist
        wl['temp_vol'] = [units.parse(v).value for v in wl["volume"]]
        wl.loc[wl['stock'].str.contains('placeholder'), 'stock'] = 'water'
        wl_new = wl.groupby(['well', 'stock'], as_index=False)['temp_vol'].sum()
        wl_new['volume'] = [units.parse(f"{v} ul") for v in wl_new['temp_vol']]
        wl_new = wl_new.drop('temp_vol', axis=1)
        file.worklist = wl_new

    # Check for low-volume dispenses
    print("\nChecking low dispense volumes...")
    n_tot = 0
    for plate, file in instructions[0].plates:
        wl = file.worklist
        low_dispenses = [round(v.value, 1) for v in wl["volume"].tolist()]
        low_dispenses = [v for v in low_dispenses if v < 10.0]
        n_low = len(low_dispenses)
        n_tot += n_low
        print(f"  {plate.plate_id}: {n_low} dispenses under 10 µL")

    tip_box_for_low_vol = int(round(n_tot / 96, 0))
    tip_box_for_inoc = len(instructions[0].plates)

    # Add first round of tip boxes for low volume dispensing
    for n in range(tip_box_for_low_vol):
        with open(labware_path, 'a') as f:
            line = f"tips{n+1}\tFCA, 50ul Filtered\t{wt['tip_box']['name']}\t{wt['tip_box']['current']}\n"
            f.write(line)
        wt['tip_box']['current'] += 1

    print("\nHigh-level experimental needs...")
    print(f"Tip boxes for low volume dispenses: {tip_box_for_low_vol} (n = {n_tot})")
    print(f"Tip boxes for inoculation: {tip_box_for_inoc}")

    # calculate total media needs, assuming constant addition to each well
    env_counts = layout.groupby('environment')['plate'].apply(lambda x: x.unique().tolist()).apply(len).to_dict()
    for env_cond, count in env_counts.items():
        mv = (tv - iv) / media_x
        mv_ml = units.parse(f"{mv} l").convert('ml').value
        total_media = mv_ml * 96 * count
        print(f"Media ({media_x}X) for environment {env_cond} ({count} plates): {total_media:.2f} ml")
        
        # Add media to labware list
        with open(labware_path, 'a') as f:
            line = f"media_{env_cond}\t100ml\t{wt['trough']['name']}\t{wt['trough']['current']}\n"
            f.write(line)
        wt['trough']['current'] += 1
    
    # Create stock list. Total volumes are based on experiment needs alone
    stock_list = tecan.make_stock_list(
        instruction_set, 
        layout, 
        inoculation_volume=str(inoc_vol),
        min_volume="10 ul",
        single_excess_volume = '0 ul',
    )

    # Generate multi-concentration stock dilution worklist
    # NOTE: this requires more volume of the 20X stocks and will output a modified stock list
    multistock_list, stock_list_adj = tecan.make_multistock_worklist(
        stock_list, 
        container=stock_containers,
        excess=fill_excess,
        multi_excess_volume = 0.05,  # 5% of total volume as excess for multistock fills
    )

    # Filter and print only single-dispense stocks and 20x stocks
    stock_list_adj_copy = stock_list_adj.copy()
    stock_list_adj_copy['core_name'] = stock_list_adj_copy['stock'].str.replace(r' x\d+$', '', regex=True)
    stock_list_adj_copy['stock_x'] = stock_list_adj_copy['stock'].str.extract(r' x(\d+\.?\d*)$')[0]
    stock_list_adj_copy['num_stocks'] = stock_list_adj_copy.groupby('core_name')['core_name'].transform('size')
    
    # Filter: single stocks (num_stocks == 1) OR stocks with "x20" or "x20.0"
    filtered_stocks = stock_list_adj_copy[
        (stock_list_adj_copy['num_stocks'] == 1) | 
        (stock_list_adj_copy['stock_x'].isin(['20', '20.0']))
    ].copy()
    
    # Print the list of stocks a person needs to prepare
    for idx, row in filtered_stocks.iterrows():
        print(f"{row['stock']}: {row['total_volume_needed']}")

    # Add 20X stocks to labware list
    x20_stocks = stock_list_adj_copy[
        (stock_list_adj_copy['stock_x'].isin(['20', '20.0']))
    ].copy()
    
    for idx, row in x20_stocks.iterrows():
        stock_name = row['stock']
        with open(labware_path, 'a') as f:
            line = f"{stock_name}\t50ml Falcon\t{wt['tube']['name']}\t{wt['tube']['current']}\n"
            f.write(line)
        wt['tube']['current'] += 1

    # create stock map of the other stocks
    stock_map = copy.deepcopy(multistock_list)
    stock_map = stock_map[["DestinationLabwareID", "DestinationPosition", "Volume", "TargetStock"]]
    other_totals = stock_map.groupby(['TargetStock', 'DestinationLabwareID', 'DestinationPosition']).agg({
        'Volume': 'sum'
    }).reset_index()

    # Create a total stock map
    other_totals = other_totals.rename(columns={'DestinationLabwareID': 'LabwareID', 'DestinationPosition': 'Position', 'TargetStock': 'Stock'})
    
    orange_g_stock = pd.DataFrame({'LabwareID': ['o2_and_orange_g'], 'Position': ['A1'], 'Volume': [100], 'Stock': ['orange_g x50']})
    
    # O2-limitation stock split along 8 rows
    single_stocks = filtered_stocks[filtered_stocks['num_stocks'] == 1].copy()
    o2_row = single_stocks[single_stocks['stock'].str.contains('o2', case=False, na=False)]
    vol = o2_row.iloc[0]['total_volume_needed']
    o2_total_ul = units.parse(vol).convert('ul').value
    o2_ul_per_well = round(o2_total_ul / 8)
    o2_expanded = []
    row_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for row_letter in row_letters:
        o2_expanded.append({
            'LabwareID': 'o2_and_orange_g',
            'Position': f"{row_letter}2",
            'Volume': o2_ul_per_well,
            'Stock': 'o2 x20.0'
        })
    o2_stock = pd.DataFrame(o2_expanded)

    all_stock_map = other_totals.copy()

    # Sort multistock list: water first, then by target stock (descending), 
    # destination labware (descending), row letter (ascending), and column number (descending)
    multistock_list['row_letter'] = multistock_list['DestinationPosition'].str[0]
    multistock_list['col_num'] = multistock_list['DestinationPosition'].str[1:].astype(int)
    multistock_list['is_fill'] = multistock_list['SourceLabwareID'] == fill_excess
    
    multistock_list = multistock_list.sort_values(
        by=['is_fill', 'SourceLabwareID', 'DestinationLabwareID', 'col_num', 'row_letter'],
        ascending=[False, True, True, False, True]
    ).reset_index(drop=True)
    
    multistock_list = multistock_list.drop(['col_num', 'row_letter', 'is_fill'], axis=1)


    # Export stock labware to labware list, using all_stock_map
    for labware_id in all_stock_map['LabwareID'].unique():
        # Determine labware type based on stock_containers variable
        if 'conical' in stock_containers.lower():
            labware_type = 'tube'
            full_labware_type = '50mL Falcon'
        else:
            labware_type = 'plate'
            full_labware_type = '96 Well Flat_BacterAI'
        
        # Append to labware list file
        with open(labware_path, 'a') as f:
            line = f"{labware_id}\t{full_labware_type}\t{wt[labware_type]['name']}\t{wt[labware_type]['current']}\n"
            f.write(line)
        
        # Increment the current position counter
        wt[labware_type]['current'] += 1
        
    # Add additional stock labware custom: water and o2_and_orange_g
    with open(labware_path, 'a') as f:
        line = f"water\t100ml\t{wt['trough']['name']}\t{wt['trough']['current']}\n"
        f.write(line)
    wt['trough']['current'] += 1
    with open(labware_path, 'a') as f:
        line = f"o2_and_orange_g\t96 Well Flat_BacterAI\t{wt['plate_first3']['name']}\t{wt['plate_first3']['current']}\n"
        f.write(line)
    wt['plate_first3']['current'] += 1

    all_stock_map = pd.concat([all_stock_map, orange_g_stock, o2_stock], ignore_index=True)


    # Export Tecan worklists (creates the "worklists" subfolder)
    tecan.generate_experiment_files(
        instruction_set, 
        layout, 
        min_volume="10 uL",
        path_to_worklists=output_filepath, 
    )

    # Modify worklists with updated stock positions
    worklist_dir = os.path.join(output_filepath, "worklists", instruction_set.id)
    worklist_files = [f for f in os.listdir(worklist_dir) if f.endswith('.csv') and instruction_set.id in f and 'multistock' not in f]
    
    for worklist_file in worklist_files:
        worklist_path = os.path.join(worklist_dir, worklist_file)
        worklist_df = pd.read_csv(worklist_path)
        
        # Process each row in the worklist
        for idx, row in worklist_df.iterrows():
            source_labware_id = row['SourceLabwareID']
                
            # Special handling for any stock that isn't water (i.e., fill_excess)
            # we put all the stocks on multi-stock plates, so we need to update the source labware and well positions from those
            if source_labware_id != fill_excess:
                # Extract row letter from DestinationPosition
                dest_position = row['DestinationPosition']
                dest_row_letter = dest_position[0]
                # extract the well column number from the stock map -- note that this assumes one column per stock on the multi-stock plates
                stock_match = all_stock_map[all_stock_map['Stock'] == source_labware_id]
                if stock_match.empty:
                    raise ValueError(f"Error in altering worklist to use multi-stock plates: {source_labware_id} not found in stock map.")
                stock_match = stock_match.iloc[0]
                source_position = stock_match['Position']
                source_col_number = int(source_position[1:])
                # Update SourcePosition to match row letter of destination and column number of stock
                worklist_df.at[idx, 'SourcePosition'] = f"{dest_row_letter}{source_col_number}"
                # update SourceLabwareID to match the stock plate (including the x20 since those stocks are also present on the multi-stock plates)
                worklist_df.at[idx, 'SourceLabwareID'] = stock_match['LabwareID']
        
        # Save / overwrite the updated worklist
        worklist_df.to_csv(worklist_path, index=False)
    
    # Export multistock worklist
    multistock_path = os.path.join(
        output_filepath, "worklists", instruction_set.id, 
        f"{instruction_set.id}_multistock_worklist.csv"
    )
    multistock_list.to_csv(multistock_path, index=False)
    # print(f"  - {instruction_set.id}_multistock_worklist.csv (stock dilution worklist)")

    # Export experiment map (creates the "plate_maps" subfolder)
    mapper.save_well_map(
        path=output_filepath,
        layout=layout,
        instruction_set_id=instruction_set.id,
        dimensions=constants.labware["WP96_325ul_no_corner"].shape,
    )

    #Export full stock map
    all_stock_map_path = os.path.join(
        output_filepath, "plate_maps", instruction_set.id,  
        f"{instruction_set.id}_stock_map.csv"
    )
    all_stock_map.to_csv(all_stock_map_path, index=False)

    # How many containers for the on-demand stocks?
    ms_map = copy.deepcopy(multistock_list)
    ms_map = ms_map[["TargetStock", "DestinationLabwareID", "DestinationPosition"]]
    ms_map = ms_map.drop_duplicates(subset=["DestinationLabwareID", "DestinationPosition"])

    print(f"\nFree, on-demand stock containers needed on worktable to begin: {len(ms_map['DestinationLabwareID'].unique())}")
    # print(f"  - {instruction_set.id}_on_demand_stock_map.csv (stock location map)")


    # Export plate-to-file tracking
    ptf = []
    for plate in instructions[0].plates:
        lo = layout.loc[layout['plate'] == plate[0].plate_id]
        ids = pd.DataFrame({
            'parent_plate': [plate[0].plate_id],
            'file_id': [plate[1].file_id],
            'environment': lo['environment'].iloc[0],
        })
        ptf.append(ids)
    
    ptf = pd.concat(ptf, ignore_index=True)
    ptf_filepath = os.path.join(
        output_filepath, "plate_maps", instruction_set.id,
        f"{instruction_set.id}_plate_to_file_id.csv"
    )
    ptf.to_csv(ptf_filepath, index=False)
    # print("\n Plate ID tracking: plate_to_file_id.csv")

    # Create location for instrument data
    data_path = os.path.join(output_filepath, "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # NOTE: All files placed in the "export_to_tecan" folder are named with the unique instruction set ID hashcode
    # as such, it should be impossible to overwrite files from different experiments, and they could in theory be placed
    # into the same folder on the Tecan computer.


    # Generate GWL worklists
    
    # Calculate media volume per well (in µL)
    mv = (tv - iv) / media_x
    mv_ul = units.parse(f"{mv} l").convert('ul').value

    dispense_media = []
    
    # 1. Generate reagent distribution worklist for media
    for env_cond in ptf['environment'].unique():
        env_plates = ptf[ptf['environment'] == env_cond]['file_id'].tolist()
        source_name = f"media_{env_cond}"
        
        # Generate GWL for this environment's media distribution
        media_gwl = tecan.make_reagent_dispense_gwl(
            instructions=instructions,
            SrcRackLabel=source_name,
            SrcRackType='100ml',
            DestRackLabel=env_plates,
            DestRackType='96 Well Flat',
            Volume=mv_ul,
            LiquidClass='Water Free Multi',
            with_lids=True,
        )

        dispense_media.extend(media_gwl)
        
    # Export media distribution GWL
    media_gwl_path = os.path.join(
        worklist_dir, 
        f"{instruction_set.id}_media_distribute.gwl"
    )

    with open(media_gwl_path, 'w') as f:
        for line in dispense_media:
            f.write(line + '\n')
    
    
    # 2. Generate GWL for multistock worklist
    multistock_gwl = tecan.make_gwl(
        multistock_list,
        instructions=instructions,
        SourceRackLabel='SourceLabwareID',
        SourcePosition='SourcePosition',
        RackLabel='DestinationLabwareID',
        RackType='96 Well Flat',
        Position='DestinationPosition',
        Volume='Volume',
        LiquidClass='Water Free Multi',
        multi=True,
    )

    # Modify labware types in multistock GWL -- 20X and water do not start out on plates
    for line in range(len(multistock_gwl)):
        if multistock_gwl[line].startswith('A;'):
            if 'water' in multistock_gwl[line]:
                parts = multistock_gwl[line].split(';')
                parts[3] = '100ml'
                parts[4] = '1' # adjust position A1 to the numeric value of 1
                repl_line = ';'.join(parts)
            elif ' x20' in multistock_gwl[line]:
                parts = multistock_gwl[line].split(';')
                parts[3] = '50 ml Falcon'
                parts[4] = '1'
                repl_line = ';'.join(parts)
            multistock_gwl[line] = repl_line
    
    # Export multistock GWL
    multistock_gwl_path = os.path.join(
        worklist_dir,
        f"{instruction_set.id}_multistock_worklist.gwl"
    )
    with open(multistock_gwl_path, 'w') as f:
        f.write('\n'.join(multistock_gwl))
    
    
    # 3. Generate GWL for regular worklists
    for worklist_file in worklist_files:
        worklist_path = os.path.join(worklist_dir, worklist_file)
        worklist_df = pd.read_csv(worklist_path)
        
        # Generate GWL
        regular_gwl = tecan.make_gwl(
            worklist_df,
            instructions=instructions,
            SourceRackLabel='SourceLabwareID',
            SourcePosition='SourcePosition',
            RackLabel='DestinationLabwareID',
            RackType='96 Well Flat',
            Position='DestinationPosition',
            Volume='Volume',
            LiquidClass='Water Free Single',
            multi=False,
            with_lids=True,
        )

        # Modify labware types in GWL -- 20X and water are not in plates
        for line in range(len(regular_gwl)):
            if regular_gwl[line].startswith('A;'):
                if 'water' in regular_gwl[line]:
                    parts = regular_gwl[line].split(';')
                    parts[3] = '100ml'
                    repl_line = ';'.join(parts)
                    regular_gwl[line] = repl_line
                elif ' x20' in regular_gwl[line]:
                    parts = regular_gwl[line].split(';')
                    parts[3] = '50 ml Falcon'
                    repl_line = ';'.join(parts)
                    regular_gwl[line] = repl_line
        
        # Export GWL
        gwl_filename = worklist_file.replace('.csv', '.gwl')
        gwl_path = os.path.join(worklist_dir, gwl_filename)
        with open(gwl_path, 'w') as f:
            f.write('\n'.join(regular_gwl))


    # Next-to-last step -- we need to change the worklist (not the multi-stock creation, but the add-to-real platesd worklists)
    # to reflect the new locations of the stock plates as listed in the stock map CSV



    # Create location for lists to export to Tecan
    tecan_path = os.path.join(output_filepath, "export_to_tecan")
    if not os.path.exists(tecan_path):
        os.makedirs(tecan_path)

    # copy the relevant plate maps to the export_to_tecan folder
    platemap_src = os.path.join(output_filepath, "plate_maps", instruction_set.id)
    platemap_list = [f for f in os.listdir(platemap_src) if f.endswith('.csv') and instruction_set.id in f]

    for map_file in platemap_list:
        platemap_src = os.path.join(
            output_filepath, "plate_maps", instruction_set.id,
            map_file
        )
        platemap_dst = os.path.join(tecan_path, map_file)
        shutil.copyfile(platemap_src, platemap_dst)

    # copy the labware list to the export_to_tecan folder
    labware_list_src = os.path.join(
        output_filepath, "plate_maps", instruction_set.id,
        f"{instruction_set.id}_labware_list.txt"
    )
    labware_list_dst = os.path.join(tecan_path, f"{instruction_set.id}_labware_list.txt")
    shutil.copyfile(labware_list_src, labware_list_dst)
        
    # copy the GWL worklists to the export_to_tecan folder
    worklist_src = os.path.join(output_filepath, "worklists", instruction_set.id)
    worklist_list = [f for f in os.listdir(worklist_src) if f.endswith('.gwl') and instruction_set.id in f]

    for gwl_file in worklist_list:
        gwl_src = os.path.join(
            output_filepath, "worklists", instruction_set.id,
            gwl_file
        )
        gwl_dst = os.path.join(tecan_path, gwl_file)
        shutil.copyfile(gwl_src, gwl_dst)
    

    print(f"\n{'='*60}")
    print(f"Location:")
    print(f"  {output_filepath}")
    print(f"\nInstruction set ID: {exp_id}")
    print(f"Date generated: {dtt}")
    print(f"{'='*60}\n")
    
    return instruction_set, layout, output_filepath



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PlatePlan experiment scheduler for Tecan liquid handler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --round 1 --total-volume "274 ul" --inoculation-volume "2 ul" --media_x 4 --prefix "PputidaExp"
  %(prog)s -r 2 -t "300 ul" -i "5 ul" -m 10 -p "PputidaExp"

Note:
  Working volume is calculated as: total - inoculation - (total / media_x)
  This reserves space for both inoculation and stock media.
        """
    )
    
    parser.add_argument(
        "-r", "--round",
        type=int,
        required=True,
        help="Experiment round number (e.g., 1, 2, 3)"
    )
    
    parser.add_argument(
        "-t", "--total-volume",
        type=str,
        required=True,
        help='Total volume per well (e.g., "274 ul", "300 ul")'
    )
    
    parser.add_argument(
        "-i", "--inoculation-volume",
        type=str,
        required=True,
        help='Inoculation volume per well (e.g., "2 ul", "5 ul")'
    )
    
    parser.add_argument(
        "-m", "--media-x",
        type=int,
        required=True,
        help="Media concentration multiplier (e.g., 4 for 4X media, 10 for 10X media)"
    )
    
    parser.add_argument(
        "-p", "--prefix",
        type=str,
        required=True,
        help="Short experiment prefix for identification (e.g., 'Round1', 'Exp2')"
    )
    
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Base directory path (default: current working directory)"
    )
    
    args = parser.parse_args()
    
    # Run main function
    # If needed, specify new stock-X concentrations from on line: 
    main(
        round_number=args.round,
        total_volume=args.total_volume,
        inoculation_volume=args.inoculation_volume,
        media_x=args.media_x,
        experiment_prefix=args.prefix,
        base_path=args.base_path,
        strains={"PpJE3959": "P_putida_JE3959"},
        environments={'pH5': 'pH::5', 'pH7': 'pH::7', 'pH9': 'pH::9'},  # MOPS buffer = ph 7, MES = pH 5, TAPS = pH 9
        stock_x_range=[2, 4, 6, 10, 14, 20],
        stock_containers="WP96_325ul",  # e.g., dRP_21ml, WP96_325ul, Conical50ml
    )
# If you made changes to plateplan, reinstall with
# cd ~/PlatePlan_pnnl
# conda activate plateplan
# pip install -e ./src/plateplan
# cd your/experiment/path