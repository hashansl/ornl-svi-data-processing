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
from persim.images_weights import linear_ramp


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
    
    # print(f'selected census: {selected_census}')

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

    # print(f"Number of connected components: {n_components}")

    # get the simplices for each component(network)
    component_census = {i: [] for i in range(n_components)}  # Initialize a dictionary for simplices per component
    component_simplices = {i: [] for i in range(n_components)}  # Initialize a dictionary for simplices per component

    # Get the index of the selected census(this way missing census(not selected) will not be included)
    id = QTemp.index.to_list()

    # if there are multiple components in the graph. Assign the simplices to the corresponding component
    

    for label, idx in zip(labels, id):
        component_census[label].append(idx)
    
    for simplex in simplices:
        if len(simplex) == 2 or len(simplex) == 3:
            # take the first vertice in the simplex and check component census it belongs to
            vertice = simplex[0]
            for component in component_census:
                
                if vertice in component_census[component]:
                    # print(f'vertice {vertice} belongs to component {component}')
                    component_simplices[component].append(simplex)


    data_frame[variable_name+'_marginal_variance'] = None #delete this line

    # assign generalized variance for each n_component
    generalized_variance_dic = {i: [] for i in range(n_components)}  # Initialize a dictionary for each n_component

    for k in range(n_components):
        # print(k)

        # get the length of the labels array where the value is equal to i
        # print(len(labels[labels == k]))

        if len(labels[labels==k])==1:

            # get the index of the label
            index = np.where(labels==k)[0][0]
            # print(index)

            #this part is not written becase: does not exists

            # # get the index from Q_df
            # print(Q_df.index[index])

            # print(f"Region {k} is an isolated region")
            # print(f"Marginal Variances with FIPS: {list(zip(Qmatrix[0].index, marginal_variances))}")
            generalized_variance_dic[k] = 1  #CHECK THIS VALUE
        else:
            # print(f"Region {k} is a connected region")

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

            generalized_variance_dic[k] = generalized_variance

            # print(f"Generalized Variance: {generalized_variance}")

    return generalized_variance_dic, component_census, component_simplices


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

        # print("County")
        # print(county_svi_df)
    
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

            # print(f'length of simplices: {len(simplices)}')

            if len(simplices)==0:
                print(f'No simplices for {variable_name} in {county_stcnty}')
                # print(df_one_variable)
            else:
                # print(f'State: {state}')
                # print(f'County: {county_stcnty}')
                # print(f'County: {variable_name}')

                # print("Simplices",simplices)

                generalized_variance_dic, component_census, component_simplices = generate_generalized_variance(simplices=simplices,data_frame=df_one_variable, variable_name=variable_name)

                # print(f'Generalized Variance: {generalized_variance_dic}\n')

                # print(f'Generalized Variance: {generalized_variance}')

                # Generate persistence images based on the generalized variance
                generate_persistence_images(simplices, df_one_variable, variable_name, county_stcnty, base_path, PERSISTENCE_IMAGE_PARAMS, generalized_variance_dic, component_census, component_simplices)

            # break

        # break


def process_county(state, county, variable, selected_variables_with_censusinfo, base_path, PERSISTENCE_IMAGE_PARAMS, INFINITY):

    svi_od_path = os.path.join(data_path, state, state + '.shp')
    svi_od = gpd.read_file(svi_od_path)
    # # for variable in selected_variables:
    #     # svi_od = svi_od[svi_od[variable] != -999]

        
    svi_od_filtered_state = svi_od[selected_variables_with_censusinfo].reset_index(drop=True)
    county_svi_df = svi_od_filtered_state[svi_od_filtered_state['STCNTY'] == county]

    # print("County")
    # print(county_svi_df)

    df_one_variable = county_svi_df[['STCNTY','FIPS', variable, 'geometry']]
    df_one_variable = df_one_variable.sort_values(by=variable)
    df_one_variable['sortedID'] = range(len(df_one_variable))
    df_one_variable = gpd.GeoDataFrame(df_one_variable, geometry='geometry')
    df_one_variable.crs = "EPSG:3395"

    adjacencies_list, adjacent_counties_df, county_list = generate_adjacent_counties(df_one_variable, variable)
    adjacent_counties_dict = dict(zip(adjacent_counties_df['county'], adjacent_counties_df['adjacent']))
    county_list = adjacent_counties_df['county'].tolist()
    simplices = form_simplicial_complex(adjacent_counties_dict, county_list)

    # print(f'length of simplices: {len(simplices)}')

    if len(simplices)==0:
        print(f'No simplices for {variable} in {county}')
        # print(df_one_variable)
    else:
        # print(f'State: {state}')
        # print(f'County: {county_stcnty}')
        # print(f'County: {variable_name}')

        # print("Simplices",simplices)

        generalized_variance_dic, component_census, component_simplices = generate_generalized_variance(simplices=simplices,data_frame=df_one_variable, variable_name=variable)

        print(f'Generalized Variance: {generalized_variance_dic}\n')

        # print(f'Generalized Variance: {generalized_variance}')

        # Generate persistence images based on the generalized variance
        generate_persistence_images(simplices, df_one_variable, variable, county, base_path, PERSISTENCE_IMAGE_PARAMS, generalized_variance_dic, component_census, component_simplices)


def plot_persistence_image_from_dim0_and_dim1(pdgms_dim0, pdgms_dim1, PERSISTENCE_IMAGE_PARAMS, generalized_variance, save_path,save_as_image=False,save_as_numpy=False,return_raw=False):

    # print('Plotting persistence image')
    # print(f'length of dim0: {len(pdgms_dim0)}')
    # print(f'length of dim1: {len(pdgms_dim1)}')


    if len(pdgms_dim0)>0 and len(pdgms_dim1)==0:

        pimgr = PersistenceImager(pixel_size=0.01)
        pimgr.fit(pdgms_dim0)

        pimgr.pixel_size = PERSISTENCE_IMAGE_PARAMS['pixel_size']
        pimgr.birth_range = PERSISTENCE_IMAGE_PARAMS['birth_range']
        pimgr.pers_range = PERSISTENCE_IMAGE_PARAMS['pers_range']
        pimgr.kernel_params =  {'sigma': np.sqrt(generalized_variance)}

        pimgs = pimgr.transform(pdgms_dim0)
        pimgs = np.rot90(pimgs, k=1) 

        print(f'Persistence image shape: {pimgs.shape}')
        print(f'Max value: {np.max(pimgs)}')

        # get thefirst colmn of the persistence image
        A = pimgs[:,0]

        print(f'First column of the persistence image: {A}')
        print(f'Max value of the first column: {np.max(A)}')


        # save
        if save_as_image:
            plt.figure(figsize=(2.4, 2.4))
            plt.imshow(pimgs, cmap='viridis')  # Assuming 'viridis' colormap, change as needed
            plt.axis('off')  # Turn off axis
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust

            plt.savefig(save_path, dpi=300)
        elif save_as_numpy:
            np.save(save_path, pimgs)
        elif return_raw:
            return pimgs

    elif len(pdgms_dim0)>0 and len(pdgms_dim1)>0:

        pimgr_dim0 = PersistenceImager(pixel_size=0.01)
        pimgr_dim0.fit(pdgms_dim0)

        pimgr_dim0.pixel_size = PERSISTENCE_IMAGE_PARAMS['pixel_size']
        pimgr_dim0.birth_range = PERSISTENCE_IMAGE_PARAMS['birth_range']
        pimgr_dim0.pers_range = PERSISTENCE_IMAGE_PARAMS['pers_range']
        pimgr_dim0.kernel_params =  {'sigma': np.sqrt(generalized_variance)}

        pimgs_dim0 = pimgr_dim0.transform(pdgms_dim0)
        pimgs_dim0 = np.rot90(pimgs_dim0, k=1)

        pimgr_dim1 = PersistenceImager(pixel_size=0.01)
        pimgr_dim1.fit(pdgms_dim1)

        pimgr_dim1.pixel_size = PERSISTENCE_IMAGE_PARAMS['pixel_size']
        pimgr_dim1.birth_range = PERSISTENCE_IMAGE_PARAMS['birth_range']
        pimgr_dim1.pers_range = PERSISTENCE_IMAGE_PARAMS['pers_range']
        pimgr_dim1.kernel_params =  {'sigma': np.sqrt(generalized_variance)}

        # # delete this lines
        # pimgr_dim1.weight = linear_ramp
        # pimgr_dim1.weight_params = {'low':0.0, 'high':1.0, 'start':0.0, 'end':40.0}

        pimgs_dim1 = pimgr_dim1.transform(pdgms_dim1)
        pimgs_dim1 = np.rot90(pimgs_dim1, k=1)

        # get the max value of the dim0 and dim1
        max_dim0 = np.max(pimgs_dim0)
        max_dim1 = np.max(pimgs_dim1)

        # min max scaling -- to get the same scale for both dim0 and dim1
        pimgs_dim0 = (pimgs_dim0 - 0) / (max_dim0 - 0)
        pimgs_dim1 = (pimgs_dim1 - 0) / (max_dim1 - 0)

        # concatenate the dim0 and dim1 to get the final persistence image
        pimgs = pimgs_dim0 + pimgs_dim1

        # save
        if save_as_image:
            plt.figure(figsize=(2.4, 2.4))
            plt.imshow(pimgs, cmap='viridis')  # Assuming 'viridis' colormap, change as needed
            plt.axis('off')  # Turn off axis
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust

            plt.savefig(save_path, dpi=300)
        elif save_as_numpy:
            np.save(save_path, pimgs)
        elif return_raw:
            return pimgs

    elif len(pdgms_dim0)==0 and len(pdgms_dim1)>0:

        pimgr_dim1 = PersistenceImager(pixel_size=0.01)
        pimgr_dim1.fit(pdgms_dim0)

        pimgr_dim1.pixel_size = PERSISTENCE_IMAGE_PARAMS['pixel_size']
        pimgr_dim1.birth_range = PERSISTENCE_IMAGE_PARAMS['birth_range']
        pimgr_dim1.pers_range = PERSISTENCE_IMAGE_PARAMS['pers_range']
        pimgr_dim1.kernel_params =  {'sigma': np.sqrt(generalized_variance)}

        pimgs = pimgr.transform(pimgr_dim1)
        pimgs = np.rot90(pimgs, k=1) 

        # save
        if save_as_image:
            plt.figure(figsize=(2.4, 2.4))
            plt.imshow(pimgs, cmap='viridis')  # Assuming 'viridis' colormap, change as needed
            plt.axis('off')  # Turn off axis
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust

            plt.savefig(save_path, dpi=300)
        elif save_as_numpy:
            np.save(save_path, pimgs)
        elif return_raw:
            return pimgs



def generate_persistence_images(simplices, df_one_variable, variable_name, county_stcnty, base_path, PERSISTENCE_IMAGE_PARAMS, generalized_variance_dic, component_census, component_simplices):
    """Generate persistence images."""

    save_path = os.path.join(base_path, variable_name, county_stcnty)
    # save_path = base_path

    # create the directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if len(generalized_variance_dic)==1:

        generalized_variance = list(generalized_variance_dic.values())[0]

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

        pdgms_dim0 = [[birth, death] for birth, death in intervals_dim0 if death < np.inf]
        pdgms_dim1 = [[birth, death] for birth, death in intervals_dim1 if death < np.inf]

        # plot persistence image(save as a image or numpy array)
        plot_persistence_image_from_dim0_and_dim1(pdgms_dim0, pdgms_dim1, PERSISTENCE_IMAGE_PARAMS, generalized_variance, save_path, save_as_numpy=True)


    elif len(generalized_variance_dic)>1:

        print(f'county: {county_stcnty} is divided into {len(generalized_variance_dic)} subcomponents for variable {variable_name}')

        # each sub network will generate a separate persistence image
        per_images_per_subcomponent = []
        pdgms_per_subcomponent = []

        for key in component_census.keys():
            # print(key)

            generalized_variance = generalized_variance_dic[key]
            simplices_sub = component_simplices[key]
            census_sub = component_census[key]
            # print(f'Generalized Variance: {generalized_variance}')
            # print(f'Simplices: {simplices_sub}')
            # print(f'Census: {census_sub}')

            # Generate persistence images based on the generalized variance

            st = gudhi.SimplexTree()
            st.set_dimension(2)

            for simplex in census_sub:
                # print(simplex)
            #     # if len(simplex) == 1:
                st.insert([simplex], filtration=0.0)

            for simplex in simplices_sub:
                if len(simplex) == 2:
                    last_simplex = simplex[-1]
                    filtration_value = df_one_variable.loc[df_one_variable['sortedID'] == last_simplex, variable_name].values[0]
                    st.insert(simplex, filtration=filtration_value)

            for simplex in simplices_sub:
                if len(simplex) == 3:
                    last_simplex = simplex[-1]
                    filtration_value = df_one_variable.loc[df_one_variable['sortedID'] == last_simplex, variable_name].values[0]
                    st.insert(simplex, filtration=filtration_value)

            st.compute_persistence()
            persistence = st.persistence()

            intervals_dim0 = st.persistence_intervals_in_dimension(0)
            intervals_dim1 = st.persistence_intervals_in_dimension(1)

            pdgms_dim0 = [[birth, death] for birth, death in intervals_dim0 if death < np.inf]
            pdgms_dim1 = [[birth, death] for birth, death in intervals_dim1 if death < np.inf]

            pimgs = plot_persistence_image_from_dim0_and_dim1(pdgms_dim0, pdgms_dim1, PERSISTENCE_IMAGE_PARAMS, generalized_variance, save_path, return_raw=True)
            per_images_per_subcomponent.append(pimgs)

            plt.figure(figsize=(2.4, 2.4))
            plt.imshow(pimgs, cmap='viridis')  # Assuming 'viridis' colormap, change as needed
            plt.axis('off')  # Turn off axis
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust
            plt.savefig(save_path + f'_subcomponent_{key}.png', dpi=300)


        # min max scale each subcomponent
        per_images_per_subcomponent_scaled = []

        for pimg in per_images_per_subcomponent:
            pimg_scaled = (pimg - 0) / (np.max(pimg) - 0)
            per_images_per_subcomponent_scaled.append(pimg_scaled)

        # final_pimgr = np.sum(per_images_per_subcomponent, axis=0) # In here scaling two images are not considered
        final_pimgr = np.sum(per_images_per_subcomponent_scaled, axis=0)

        # plt.figure(figsize=(2.4, 2.4))
        # plt.imshow(final_pimgr, cmap='viridis')  # Assuming 'viridis' colormap, change as needed
        # plt.axis('off')  # Turn off axis
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust

        # plt.savefig(save_path, dpi=300)

        np.save(save_path, final_pimgr)


# Define the main function
if __name__ == "__main__":
    # Main execution

    data_path = "/home/h6x/git_projects/ornl-svi-data-processing/processed_data/SVI/SVI2018_NOT_SCALED_MISSING_REMOVED"
    base_path = "/home/h6x/git_projects/ornl-svi-data-processing/experiment_5"

    states = get_folders(data_path)

    # EP_PCI not included in the list of selected variables
    selected_variables = [
         'EP_POV','EP_UNEMP', 'EP_NOHSDP', 'EP_UNINSUR', 'EP_AGE65', 'EP_AGE17', 'EP_DISABL', 
        'EP_SNGPNT', 'EP_LIMENG', 'EP_MINRTY', 'EP_MUNIT', 'EP_MOBILE', 'EP_CROWD', 'EP_NOVEH', 'EP_GROUPQ'
    ]
    selected_variables_with_censusinfo = ['FIPS', 'STCNTY'] + selected_variables + ['geometry']

    PERSISTENCE_IMAGE_PARAMS = {
        'pixel_size': 1,
        'birth_range': (0.0, 100),
        'pers_range': (0.0, 100),
        'kernel_params': {'sigma': 0.0003}
    }

    INF_DELTA = 0.1
    # INFINITY = (PERSISTENCE_IMAGE_PARAMS['birth_range'][1] - PERSISTENCE_IMAGE_PARAMS['birth_range'][0]) * INF_DELTA
    INFINITY = 1

    # create_variable_folders(base_path, selected_variables)


    # single generalized variance
    # state = 'NY'
    # county = '36047'
    # variable = 'EP_POV'

    # multiple generalized variance
    state = 'NY'
    county = '36081'
    variable = 'EP_POV'


    process_county(state, county, variable, selected_variables_with_censusinfo, base_path, PERSISTENCE_IMAGE_PARAMS, INFINITY)

    # process_state(state, selected_variables, selected_variables_with_censusinfo, base_path, PERSISTENCE_IMAGE_PARAMS, INFINITY)


    # for state in tqdm(states, desc="Processing states"):

    #     process_state(state, selected_variables, selected_variables_with_censusinfo, base_path, PERSISTENCE_IMAGE_PARAMS, INFINITY)


    print('All states processed.')
