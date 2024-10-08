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

from pysal.lib import weights
from pysal.lib import weights


from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from scipy.linalg import solve
from scipy.sparse.linalg import spsolve
import numpy as np

import scipy as sp

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

def generate_generalized_variance(simplices,data_frame, variable_name):

    selected_census = []

    for set in simplices:
        if len(set) == 2 or len(set) == 3:
            for vertice in set:
                if vertice not in selected_census:
                    selected_census.append(vertice)
    
    print(selected_census)

    # print(data_frame.head(3))
    # print(data_frame.columns)

    filtered_census_df = data_frame.loc[data_frame["sortedID"].isin(selected_census)]

    # lattice stored in a geo-table
    wq = weights.contiguity.Queen.from_dataframe(filtered_census_df)
    neighbors_q = wq.neighbors

    QTemp = pd.DataFrame(*wq.full()).astype(int)
    QTemp = QTemp.multiply(-1)

    QTemp.index = filtered_census_df["sortedID"].values
    QTemp.columns = filtered_census_df["sortedID"].values

    # for each row in the fullMatrix dataframe sum the values in the row and take the absolute value and store in the diagonal
    for i in QTemp.index:
        QTemp.loc[i,i] = abs(QTemp.loc[i].sum())

    # print(neighbors_q)
    # print(filtered_census_df.head(3))
    # print(QTemp)


    # Marginal variance code -Multiple clusters

    # transform df to numpy array
    Q = QTemp.to_numpy()

    graph = csr_matrix(QTemp)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    print(f"Number of connected components: {n_components}")

    data_frame[variable_name+'_marginal_variance'] = None


    for k in range(n_components):
        # print(k)

        # get the length of the labels array where the value is equal to i
        print(len(labels[labels == k]))

        if len(labels[labels==k])==1:

            # get the index of the label
            index = np.where(labels==k)[0][0]
            print(index)

            #this part is not written becase: does not exists

            # # get the index from Q_df
            # print(Q_df.index[index])

            # print(f"Region {k} is an isolated region")
            # print(f"Marginal Variances with FIPS: {list(zip(Qmatrix[0].index, marginal_variances))}")
        else:
            print(f"Region {k} is a connected region")

            # get the location index to an array 
            index = np.where(labels == k)
            # print(index)

            # Extract the submatrix
            QQ = Q[np.ix_(index[0], index[0])]

            # print(QQ)

            n = QQ.shape[0]

            
            Q_jitter = QQ + sp.sparse.diags(np.ones(n)) * max(QQ.diagonal()) * np.sqrt(

                np.finfo(np.float64).eps

            )


            # inverse of precision (Q) is cov

            Q_perturbed = sp.sparse.csc_array(Q_jitter)

            b = sp.sparse.identity(n, format='csc')

            sigma = spsolve(Q_perturbed, b)


            # V \in Null(Q)

            V = np.ones(n)  # from pg. 6

            W = sigma @ V.T  # \Sigma * B in 3.17

            Q_inv = sigma - np.outer(W * solve(V @ W, np.ones(1)), W.T)

            # grabbing diag of cov gives var and

            # arithmetic mean in log-space becomes geometric mean after exp

            generalized_variance = np.exp(np.mean(np.log(np.diag(Q_inv))))  # equation in the paper use daba as 1
            # generalized_variance = np.exp(np.sum(np.log(np.diag(Q_inv))) / n) #same as above

            # print(f"Generalized Variance: {generalized_variance}")

            return generalized_variance




def generate_scaled_marginal_variance(simplices,data_frame, variable_name):

    selected_census = []

    for set in simplices:
        if len(set) == 2 or len(set) == 3:
            for vertice in set:
                if vertice not in selected_census:
                    selected_census.append(vertice)
    
    print(selected_census)

    # print(data_frame.head(3))
    # print(data_frame.columns)

    filtered_census_df = data_frame.loc[data_frame["sortedID"].isin(selected_census)]

    # lattice stored in a geo-table
    wq = weights.contiguity.Queen.from_dataframe(filtered_census_df)
    neighbors_q = wq.neighbors

    QTemp = pd.DataFrame(*wq.full()).astype(int)
    QTemp = QTemp.multiply(-1)

    QTemp.index = filtered_census_df["sortedID"].values
    QTemp.columns = filtered_census_df["sortedID"].values

    # for each row in the fullMatrix dataframe sum the values in the row and take the absolute value and store in the diagonal
    for i in QTemp.index:
        QTemp.loc[i,i] = abs(QTemp.loc[i].sum())

    # print(neighbors_q)
    # print(filtered_census_df.head(3))
    # print(QTemp)


    # Marginal variance code -Multiple clusters

    # transform df to numpy array
    Q = QTemp.to_numpy()

    graph = csr_matrix(QTemp)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    print(f"Number of connected components: {n_components}")

    data_frame[variable_name+'_marginal_variance'] = None


    for k in range(n_components):
        # print(k)

        # get the length of the labels array where the value is equal to i
        print(len(labels[labels == k]))

        if len(labels[labels==k])==1:

            # get the index of the label
            index = np.where(labels==k)[0][0]
            print(index)

            #this part is not written becase: does not exists

            # # get the index from Q_df
            # print(Q_df.index[index])

            # print(f"Region {k} is an isolated region")
            # print(f"Marginal Variances with FIPS: {list(zip(Qmatrix[0].index, marginal_variances))}")
        else:
            print(f"Region {k} is a connected region")

            # get the location index to an array 
            index = np.where(labels == k)
            # print(index)

            # Extract the submatrix
            QQ = Q[np.ix_(index[0], index[0])]

            # print(QQ)

            n = QQ.shape[0]

            
            Q_jitter = QQ + sp.sparse.diags(np.ones(n)) * max(QQ.diagonal()) * np.sqrt(

                np.finfo(np.float64).eps

            )


            # inverse of precision (Q) is cov

            Q_perturbed = sp.sparse.csc_array(Q_jitter)

            b = sp.sparse.identity(n, format='csc')

            sigma = spsolve(Q_perturbed, b)


            # V \in Null(Q)

            V = np.ones(n)  # from pg. 6

            W = sigma @ V.T  # \Sigma * B in 3.17

            Q_inv = sigma - np.outer(W * solve(V @ W, np.ones(1)), W.T)

            # grabbing diag of cov gives var and

            # arithmetic mean in log-space becomes geometric mean after exp

            scaling = np.exp(np.sum(np.log(np.diag(Q_inv))) / n)

            # scaling_factor.append(scaling)

            # print(f"Scaling/GV: {scaling}")

            marginal_variances = np.diag(Q_inv/scaling)
            # print(f"Marginal Variances: {marginal_variances}")

            # print(f"Marginal Variances with FIPS: {list(zip(Qmatrix[0].index[index[0]], marginal_variances))}")


            # # get the Q_df index 
            # print(Q_df.index[index[0]])

            # fill the new column with the marginal variances only matching the (Q_df.index[index[0]]
            for sortedID, marginal_variance in zip(QTemp.index[index[0]], marginal_variances):
                data_frame.loc[data_frame['sortedID'] == sortedID, variable_name + '_marginal_variance'] = marginal_variance
            
            # print(data_frame[['sortedID',variable_name+'_marginal_variance']])

    # turn the column into float
    data_frame[variable_name + '_marginal_variance'] = data_frame[variable_name + '_marginal_variance'].astype('float64')

    print(data_frame)



def process_state(state, selected_variables, selected_variables_with_censusinfo, base_path, PERSISTENCE_IMAGE_PARAMS, INFINITY):
    """Process data for a given state."""
    svi_od_path = os.path.join(data_path, state, state + '.shp')
    svi_od = gpd.read_file(svi_od_path)
    # # for variable in selected_variables:
    #     # svi_od = svi_od[svi_od[variable] != -999]

        
    svi_od_filtered_state = svi_od[selected_variables_with_censusinfo].reset_index(drop=True)

    # Get the unique counties
    unique_county_stcnty = svi_od_filtered_state['STCNTY'].unique()

    for county_stcnty in unique_county_stcnty:
        # Filter the dataframe to include only the current county
        county_svi_df = svi_od_filtered_state[svi_od_filtered_state['STCNTY'] == county_stcnty]

        print("County")
        print(county_svi_df)
    
        for variable_name in selected_variables:
            df_one_variable = county_svi_df[['STCNTY','FIPS', variable_name, 'geometry']]
            df_one_variable = df_one_variable.sort_values(by=variable_name)
            df_one_variable['sortedID'] = range(len(df_one_variable))
            df_one_variable = gpd.GeoDataFrame(df_one_variable, geometry='geometry')
            df_one_variable.crs = "EPSG:3395"

            adjacencies_list, adjacent_counties_df, county_list = generate_adjacent_counties(df_one_variable, variable_name)
            adjacent_counties_dict = dict(zip(adjacent_counties_df['county'], adjacent_counties_df['adjacent']))
            county_list = adjacent_counties_df['county'].tolist()
            simplices = form_simplicial_complex(adjacent_counties_dict, county_list)

            print(f'State: {state}')
            print(f'County: {county_stcnty}')
            print(f'County: {variable_name}')

            # print(simplices)

            generalized_variance = generate_generalized_variance(simplices=simplices,data_frame=df_one_variable, variable_name=variable_name)

            print(f'Generalized Variance: {generalized_variance}')

            



            break

        break



# Define the main function
if __name__ == "__main__":
    # Main execution
    base_path = '/home/h6x/git_projects/data_processing/processed_data/adjacency_pers_images_npy_county/experimet_4/npy_all_variables'
    # data_path = '/home/h6x/git_projects/data_processing/processed_data/SVI/SVI2018_MIN_MAX_SCALED_MISSING_REMOVED'
    data_path = '/home/h6x/git_projects/ornl-svi-data-processing/processed_data/SVI/SVI2018_MIN_MAX_SCALED_MISSING_REMOVED'

    states = get_folders(data_path)

    # EP_PCI not included in the list of selected variables
    selected_variables = [
         'EP_POV','EP_UNEMP', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 
        'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ'
    ]
    selected_variables_with_censusinfo = ['FIPS', 'STCNTY'] + selected_variables + ['geometry']

    # PERSISTENCE_IMAGE_PARAMS = {
    #     'pixel_size': 0.001,
    #     'birth_range': (0.0, 1.00),
    #     'pers_range': (0.0, 0.40),
    #     'kernel_params': {'sigma': 0.0003}
    # }

    PERSISTENCE_IMAGE_PARAMS = {
            'pixel_size': 0.01,
            'birth_range': (-7.0, 0.00),
            'pers_range': (0.0, 3.00),
            'kernel_params': {'sigma': 0.008}
        }

    INF_DELTA = 0.1
    # INFINITY = (PERSISTENCE_IMAGE_PARAMS['birth_range'][1] - PERSISTENCE_IMAGE_PARAMS['birth_range'][0]) * INF_DELTA
    INFINITY = 1

    # create_variable_folders(base_path, selected_variables)

    for state in tqdm(states, desc="Processing states"):

        process_state(state, selected_variables, selected_variables_with_censusinfo, base_path, PERSISTENCE_IMAGE_PARAMS, INFINITY)
        break

    print('All states processed.')
