import pandas as pd
import csv
import matplotlib.pyplot as plt
from scipy.stats import linregress
from itertools import combinations
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/Users/juar705/dashboard/bacterAI_dash/P_putida_AG5577_baseline/Round1/mapped_data_2025-04-15_biotek_final_od_data.csv')
ingredient_columns = ['d_glucose', 'sodium_acetate', 'sodium_citrate', 'sodium_octanoate', 'sodium_benzoate']
growth_column = 'y'
df = df.dropna(subset=ingredient_columns + [growth_column]).reset_index(drop=True)
od600 = df['y'].values
conversion_factors = {
    "glucose": 0.5,
    "sodium_octanoate": 0.8,
    "sodium_benzoate": 0.25,
    "sodium_citrate": 0.38,
    "sodium_acetate": 0.27,
}
# mmol/gDW/hr
uptake_rates = {
    "glucose": -4.0,
    "sodium_octanoate": -3.0,
    "sodium_benzoate": -3.45,
    "sodium_citrate": -4.0,
    "sodium_acetate": -12.0,
}
yield_coefficients = {
    "glucose": 0.39,
    "sodium_octanoate": 1.08,
    "sodium_benzoate": 0.896,
    "sodium_citrate": 0.898,
    "sodium_acetate": 0.699,
}

# final concentration in well (mmol/L)
substrate_concentrations = {
    "glucose": df["d_glucose"].values,
    "sodium_acetate": df["sodium_acetate"].values,
    "sodium_citrate": df["sodium_citrate"].values,
    "sodium_octanoate": df["sodium_octanoate"].values,
    "sodium_benzoate": df["sodium_benzoate"].values,
}


carbon_sources = ['glucose', 'sodium_octanoate', 'sodium_benzoate', 'sodium_citrate', 'sodium_acetate']

def compute_modeled_od600(df, carbon_sources, conversion_factors, uptake_rates, yield_coefficients, combinations, substrate_concentrations):
    """
    Compute modeled OD600 for each carbon source in the dataframe using substrate concentrations.
    Adds a new column for each modeled OD600.
    Also computes and prints the regression slope/intercept between OD600 and biomass for each source.
    """
    cdw_dict = {}
    biomass_dict = {}
    regression_results = {}

    for source in carbon_sources:
        substrate_conc = substrate_concentrations[source]
        conversion_factor = conversion_factors[source]
        uptake_rate = uptake_rates[source]
        yield_coefficient = yield_coefficients[source]
        
        # Calculate biomass produced from substrate
        biomass = substrate_conc * yield_coefficient * uptake_rate

        # get slopes of od 600 and biomass using regression
        slope, intercept, r_value, p_value, std_err = linregress(biomass, od600)
        regression_results[source] = {'slope': slope, 'intercept': intercept}
        print(f"{source}: slope={slope:.4f}, intercept={intercept:.4f}")
        
        # convert biomass to OD600
        modeled_od600 = slope * biomass + intercept
        df[f'modeled_od600_{source}'] = modeled_od600
        cdw_dict[source] = substrate_conc * conversion_factor
        biomass_dict[source] = biomass
    
    for combo in all_combinations:
        combo_name = '+'.join(combo)
        modeled_ods = np.stack([df[f'modeled_od600_{s}'] for s in combo], axis=1) 
        # asssuming best growth scenario is the substrate who contributes the most to od600
        #sum
        df[f'modeled_od600_{combo_name}_max'] = np.sum(modeled_ods, axis=1)

    return df

all_combinations = []
for r in range(2, len(carbon_sources) + 1):
    all_combinations.extend(combinations(carbon_sources, r))

for combo in all_combinations:
    print(combo)

df = compute_modeled_od600(df, carbon_sources, conversion_factors, uptake_rates, yield_coefficients, all_combinations, substrate_concentrations)

output_columns = (
    ['y'] +
    [f'modeled_od600_{s}' for s in carbon_sources] +
    [f'modeled_od600_{"+".join(combo)}_max' for combo in all_combinations]
)
df_modeled = df[output_columns]

#regression fit modeled per ingredient
ingredient_name = 'glucose'
x = df['y']
y = df[f'modeled_od600_{ingredient_name}']
slope, intercept, r_value, p_value, std_err = linregress(x, y)
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = slope * x_fit + intercept

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='skyblue', label='Regression Method (data)')
plt.plot(x_fit, y_fit, color='red', label='Regression Line (fit)', linestyle='dashed')
plt.xlabel("Experimental OD600")
plt.ylabel("Modeled OD600 (Regression)")
plt.title(f"Experimental vs Modeled OD600: {ingredient_name} Regression Model and Fit")
plt.legend()
plt.show()

#regression fit modeled in combinations
combo_name = 'glucose+sodium_octanoate+sodium_benzoate+sodium_citrate+sodium_acetate'
x = df['y']
y = df[f'modeled_od600_{combo_name}_max']

slope, intercept, r_value, p_value, std_err = linregress(x, y)
x_fit = np.linspace(x.min(), x.max(), len(od600))
y_fit = slope * x_fit + intercept

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='lightgreen', label='Sum Logic (combo)')
plt.plot(x_fit, y_fit, color='red', linestyle='dashed', label='Regression Line (fit)')
plt.xlabel("Experimental OD600")
plt.ylabel("Modeled OD600 (Sum Logic, Combo)")
plt.title(f"Experimental vs Modeled OD600: \n {combo_name}")
plt.legend()
plt.show()



