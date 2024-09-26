import os
import warnings

import geopandas as gpd
import pandas as pd
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)


def load_data(svi_path, overdose_path):
    """Load SVI and overdose data."""
    us_svi = gpd.read_file(svi_path)
    overdose_df = pd.read_excel(overdose_path)
    return us_svi, overdose_df


def preprocess_overdose_data(overdose_df):
    """Preprocess overdose data."""
    overdose_df['GEO ID'] = overdose_df['GEO ID'].astype(str)
    overdose_df['GEO ID'] = overdose_df['GEO ID'].apply(lambda x: x.zfill(5))
    return overdose_df


def preprocess_svi_data(us_svi, raw_variables):
    """Preprocess SVI data by removing invalid values and normalizing."""
    for variable in raw_variables:
        us_svi = us_svi[us_svi[variable] != -999.00]

    # for var in raw_variables:
    #     max_val = us_svi[var].max()
    #     min_val = us_svi[var].min()
    #     us_svi[var] = (us_svi[var] - min_val) / (max_val - min_val)
    
    return us_svi


def get_states(us_svi):
    """Get unique state abbreviations excluding DC."""
    states = us_svi['ST_ABBR'].unique()
    states = states[states != 'DC']
    return states


def process_state_data(state, us_svi, overdose_df, output_dir):
    """Process data for a single state."""
    print(f'Processing: {state}')
    try:
        state_svi = us_svi[us_svi['ST_ABBR'] == state]
        # state_overdose = overdose_df[overdose_df['State Abbreviation'] == state]

        # state_overdose['Narcotic Overdose Mortality Rate 2018'] = state_overdose[
            # 'Narcotic Overdose Mortality Rate 2018'].astype(float)

        # if state == 'AK':
            # state_overdose['Narcotic Overdose Mortality Rate 2018'] += np.random.normal(0, 1e-5, state_overdose.shape[0])

        # state_overdose['percentile'] = pd.qcut(
            # state_overdose['Narcotic Overdose Mortality Rate 2018'],
            # q=[0, 0.2, 0.4, 0.6, 0.8, 1],
            # labels=['0', '1', '2', '3', '4']
        # )

        state_svi.reset_index(drop=True, inplace=True)
        # merged_df = pd.merge(state_svi, state_overdose[['GEO ID', 'percentile']],
                            #  left_on='STCNTY', right_on='GEO ID', how='left')
        # merged_df.drop(columns=['GEO ID'], inplace=True)

        gdf = gpd.GeoDataFrame(state_svi, geometry='geometry')
        # gdf['percentile'] = gdf['percentile'].astype(str)

        output_path = os.path.join(output_dir, state, f'{state}.shp')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        gdf.to_file(output_path, driver='ESRI Shapefile')

    except Exception as e:
        print(f"Error processing {state}: {e}")


def main():
    svi_path = '/home/h6x/git_projects/ornl-svi-data-processing/raw_data/svi/2018/SVI2018_US_tract.gdb'
    overdose_path = 'git_projects/ornl-svi-data-processing/raw_data/HepVu_County_Opioid_Indicators_05DEC22.xlsx'
    # output_dir = '/home/h6x/git_projects/data_processing/processed_data/SVI/2020/SVI2020_MIN_MAX_SCALED_MISSING_REMOVED'
    output_dir = '/home/h6x/git_projects/ornl-svi-data-processing/processed_data/SVI/SVI2018_NOT_SCALED_MISSING_REMOVED'

    # raw_variables = [
    #     'EP_POV', 'EP_UNEMP', 'EP_PCI', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65',
    #     'EP_AGE17', 'EP_DISABL', 'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_MUNIT',
    #     'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ'
    # ]

    raw_variables = [
        'EP_POV150', 'EP_UNEMP', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65',
        'EP_AGE17', 'EP_DISABL', 'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_MUNIT',
        'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ'
    ]

    us_svi, overdose_df = load_data(svi_path, overdose_path)
    # overdose_df = preprocess_overdose_data(overdose_df)
    us_svi = preprocess_svi_data(us_svi, raw_variables)
    states = get_states(us_svi)

    for state in states:
        process_state_data(state, us_svi, overdose_df, output_dir)


if __name__ == "__main__":
    main()
