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

def get_min_max_birth_death(pdgms):
    pdgms = np.array(pdgms)

    # count inf values
    inf_count = np.sum(np.isinf(pdgms[:, 1]))
    
    # Filter out rows where death value is infinite
    finite_pdgms = pdgms[np.isfinite(pdgms[:, 1])]
    
    if finite_pdgms.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        # Filter out rows where death value is infinite and take log
        # print(finite_pdgms)
        finite_pdgms = [[np.log(birth), np.log(death)] for birth, death in finite_pdgms if birth != 0]

        # make pdgms numpy array
        finite_pdgms = np.array(finite_pdgms)

        if finite_pdgms.size == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            # Get the min and max of birth and death from the filtered array
            min_birth = np.min(finite_pdgms[:, 0])
            max_birth = np.max(finite_pdgms[:, 0])
            min_death = np.min(finite_pdgms[:, 1])
            max_death = np.max(finite_pdgms[:, 1])

            # get the birth-death difference for each pair and get the max
            max_birth_death_diff = np.max(finite_pdgms[:, 1] - finite_pdgms[:, 0])

            
        return min_birth, max_birth, min_death, max_death, max_birth_death_diff, inf_count
    
    

def process_state(state, selected_variables, selected_variables_with_censusinfo, base_path, PERSISTENCE_IMAGE_PARAMS, INFINITY):
    """Process data for a given state."""
    svi_od_path = os.path.join(data_path, state, state + '.shp')
    svi_od = gpd.read_file(svi_od_path)
    # for variable in selected_variables:
    #     svi_od = svi_od[svi_od[variable] != -999]
    svi_od_filtered_state = svi_od[selected_variables_with_censusinfo].reset_index(drop=True)

    # Get the unique counties
    unique_county_stcnty = svi_od_filtered_state['STCNTY'].unique()

    # Create an empty list to store the new rows for the summary dataframe
    new_rows = []

    for county_stcnty in unique_county_stcnty:
        # Filter the dataframe to include only the current county
        county_svi_df = svi_od_filtered_state[svi_od_filtered_state['STCNTY'] == county_stcnty]
    
        for variable_name in selected_variables:
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

            intervals_dim1 = st.persistence_intervals_in_dimension(1)

            # print(f'Shape of the intervals_dim1: {len(intervals_dim1)}')
            # print(f'type of the intervals_dim1: {type(intervals_dim1)}')

            # print(f'Shape of the pdgms: {len(pdgms)}')
            # print(f'type of the pdgms: {type(pdgms)}')

            min_birth, max_birth, min_death, max_death, max_birth_death_diff,inf_count = get_min_max_birth_death(intervals_dim1)


            if len(intervals_dim1) > 0:

                new_rows.append({'state':state, 'county': county_stcnty,'census_c':len(df_one_variable),'variable': variable_name , 'H1':len(intervals_dim1) , 'No Persistence': "False", 'min_birth': min_birth, 'max_birth': max_birth, 'min_death': min_death, 'max_death': max_death, 'max_birth_death_diff': max_birth_death_diff, 'inf_count': inf_count})
            else:
                new_rows.append({'state':state, 'county': county_stcnty,'census_c':len(df_one_variable),'variable': variable_name , 'H1':len(intervals_dim1) , 'No Persistence': "False",  'min_birth': np.nan, 'max_birth': np.nan, 'min_death': np.nan, 'max_death': np.nan, 'max_birth_death_diff': np.nan, 'inf_count':inf_count}) # can performe per but no points

    return new_rows

# Define the main function
if __name__ == "__main__":
    # Main execution
    base_path = '/home/h6x/git_projects/data_processing/processed_data/adjacency_pers_images_npy_county/experimet_4'
    data_path = '/home/h6x/git_projects/data_processing/processed_data/SVI/SVI2018_MIN_MAX_SCALED_MISSING_REMOVED'

    states = get_folders(data_path)
    selected_variables = [
        'EP_POV', 'EP_UNEMP', 'EP_PCI', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 
        'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ'
    ]
    selected_variables_with_censusinfo = ['FIPS', 'STCNTY'] + selected_variables + ['geometry']

    PERSISTENCE_IMAGE_PARAMS = {
        'pixel_size': 0.001,
        'birth_range': (0.0, 1.0),
        'pers_range': (0.0, 1.0),
        'kernel_params': {'sigma': 0.0005}
    }

    INF_DELTA = 0.1
    INFINITY = (PERSISTENCE_IMAGE_PARAMS['birth_range'][1] - PERSISTENCE_IMAGE_PARAMS['birth_range'][0]) * INF_DELTA

    # create_variable_folders(base_path, selected_variables)

    # create an empty dataframe with the columns county, H0, H1, No Persistence
    columns = ['state','county','census_c','variable','H1','No Persistence','min_birth','max_birth','min_death','max_death','max_birth_death_diff','inf_count']
    summary_df = pd.DataFrame(columns=columns)

    for state in tqdm(states, desc="Processing states"):
        new_rows = process_state(state, selected_variables, selected_variables_with_censusinfo, base_path, PERSISTENCE_IMAGE_PARAMS, INFINITY)
        for new_row in new_rows:
            summary_df = pd.concat([summary_df, pd.DataFrame([new_row])], ignore_index=True)
        # break
    summary_df.to_csv('/home/h6x/git_projects/data_processing/processed_data/adjacency_pers_images_npy_county/experimet_4/summary_df_full_data_4.csv', index=False)

    print('All states processed.')
