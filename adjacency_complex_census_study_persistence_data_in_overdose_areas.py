# Import libraries
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import gudhi
from tqdm import tqdm
from persim import PersistenceImager
import invr
import matplotlib as mpl

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Matplotlib default settings
mpl.rcParams.update(mpl.rcParamsDefault)

# Utility functions
def get_folders(location):
    """Get list of folders in a directory."""
    return [name for name in os.listdir(location) if os.path.isdir(os.path.join(location, name))]

def generate_adjacent_counties(dataframe, variable_name):
    """Generate adjacent counties based on given dataframe and variable."""
    filtered_df = dataframe
    adjacent_counties = gpd.sjoin(filtered_df, filtered_df, predicate='intersects', how='left')
    adjacent_counties = adjacent_counties.query('sortedID_left != sortedID_right')
    adjacent_counties = adjacent_counties.groupby('sortedID_left')['sortedID_right'].apply(list).reset_index()
    adjacent_counties.rename(columns={'sortedID_left': 'county', 'sortedID_right': 'adjacent'}, inplace=True)
    adjacencies_list = adjacent_counties['adjacent'].tolist()
    county_list = adjacent_counties['county'].tolist()
    merged_df = pd.merge(adjacent_counties, dataframe, left_on='county', right_on='sortedID', how='left')
    merged_df = gpd.GeoDataFrame(merged_df, geometry='geometry')
    return adjacencies_list, merged_df, county_list

def form_simplicial_complex(adjacent_county_list, county_list):
    """Form a simplicial complex based on adjacent counties."""
    max_dimension = 3
    V = invr.incremental_vr([], adjacent_county_list, max_dimension, county_list)
    return V

def create_variable_folders(base_path, variables):
    """Create folders for each variable."""
    for variable in variables:
        os.makedirs(os.path.join(base_path, variable), exist_ok=True)
    print('Done creating folders for each variable')

def process_state(state, variable, selected_variables_with_censusinfo, base_path):
    """Process data for a given state."""
    svi_od_path = os.path.join(data_path, state, state + '.shp')
    svi_od = gpd.read_file(svi_od_path)
    # # for variable in selected_variables:
    #     # svi_od = svi_od[svi_od[variable] != -999]
 
    svi_od_filtered_state = svi_od[selected_variables_with_censusinfo].reset_index(drop=True)

    # Get the unique counties
    unique_county_stcnty = svi_od_filtered_state['STCNTY'].unique()

    # Create a empty dictionary to store the persistence data points
    persistence_data = {}

    for county_stcnty in unique_county_stcnty:
        # Filter the dataframe to include only the current county
        county_svi_df = svi_od_filtered_state[svi_od_filtered_state['STCNTY'] == county_stcnty]
    
        variable_name = variable

        df_one_variable = county_svi_df[['STCNTY', variable_name, 'geometry']]
        df_one_variable = df_one_variable.sort_values(by=variable_name)
        df_one_variable['sortedID'] = range(len(df_one_variable))
        df_one_variable = gpd.GeoDataFrame(df_one_variable, geometry='geometry')
        df_one_variable.crs = "EPSG:3395"

        adjacencies_list, adjacent_counties_df, county_list = generate_adjacent_counties(df_one_variable, variable_name)
        adjacent_counties_dict = dict(zip(adjacent_counties_df['county'], adjacent_counties_df['adjacent']))
        county_list = adjacent_counties_df['county'].tolist()
        simplices = form_simplicial_complex(adjacent_counties_dict, county_list)

        st = gudhi.SimplexTree()
        st.set_dimension(2)

        for simplex in simplices:
            if len(simplex) == 1:
                st.insert([simplex[0]], filtration=0.0)
        
        for simplex in simplices:
            if len(simplex) == 2:
                last_simplex = simplex[-1]
                filtration_value = df_one_variable.loc[df_one_variable['sortedID'] == last_simplex, variable_name].values[0]
                st.insert(simplex, filtration=filtration_value)

        for simplex in simplices:
            if len(simplex) == 3:
                last_simplex = simplex[-1]
                filtration_value = df_one_variable.loc[df_one_variable['sortedID'] == last_simplex, variable_name].values[0]
                st.insert(simplex, filtration=filtration_value)

        st.compute_persistence()
        persistence = st.persistence()

        intervals_dim0 = st.persistence_intervals_in_dimension(0)
        intervals_dim1 = st.persistence_intervals_in_dimension(1)
        pdgms = [[birth, death] for birth, death in intervals_dim1 if death < np.inf]

        # add interval dim 0  to the pdgms
        for birth, death in intervals_dim0:
            if death < np.inf:
                pdgms.append([birth, death])

            # elif death == np.inf:
                # pdgms.append([birth, INFINITY])

        # append pdgms to the persistence_data dictionary
        persistence_data[county_stcnty] = pdgms

        # save dictionary to a file
        save_path = os.path.join(base_path,f'{county_stcnty}_{variable}')
        np.save(save_path, persistence_data)

        # save_path = os.path.join(base_path, variable_name, county_stcnty)





# Define the main function
if __name__ == "__main__":
    # Main execution
    # base_path = '/home/h6x/git_projects/data_processing/processed_data/adjacency_pers_images_npy_county/2020/experimet_3/npy_all_variables'
    base_path = '/home/h6x/git_projects/ornl-svi-data-processing/processed_data/adjacency_pers_images_npy_county/persistence_data'
    data_path = '/home/h6x/git_projects/ornl-svi-data-processing/processed_data/SVI/SVI2018_MIN_MAX_SCALED_MISSING_REMOVED'

    states = get_folders(data_path)

    # EP_PCI not included in the list of selected variables
    selected_variables = [
         'EP_POV','EP_UNEMP', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 
        'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ'
    ]
    # selected_variables = [
    #      'EP_POV150','EP_UNEMP', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 
    #     'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ'
    # ]

    variable = 'EP_MOBILE'
    selected_variables_with_censusinfo = ['FIPS', 'STCNTY'] + [variable] + ['geometry']

    
    # create_variable_folders(base_path, selected_variables)

    for state in tqdm(states, desc="Processing states"):

        process_state(state, variable, selected_variables_with_censusinfo, base_path)

    print('All states processed.')