import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def get_npy_filenames(directory):
    """
    Get all .npy file names in the specified directory without the .npy extension.
    
    Args:
    directory (str): The path to the directory containing .npy files.
    
    Returns:
    list: A list of file names without the .npy extension.
    """
    file_names = []
    for file in os.listdir(directory):
        if file.endswith('.npy'):
            file_names.append(os.path.splitext(file)[0])
    return file_names


def common_files(file_dict):
    """
    Get the set of file names that are common across all lists in the given dictionary.
    
    Args:
    file_dict (dict): A dictionary where each key has a list of file names as its value.
    
    Returns:
    set: A set of file names that are common in all lists.
    """
    if not file_dict:
        return set()
    
    # Initialize the common set with the file names from the first list
    common_set = set(file_dict[next(iter(file_dict))])
    
    # Perform intersection with each subsequent list
    for key in file_dict:
        common_set.intersection_update(file_dict[key])
    
    return common_set


# Define the main function
if __name__ == "__main__":
    # Main execution

    variables = ['EP_POV','EP_UNEMP','EP_NOHSDP','EP_UNINSUR','EP_AGE65','EP_AGE17','EP_DISABL','EP_SNGPNT','EP_LIMENG','EP_MINRTY','EP_MUNIT','EP_MOBILE','EP_CROWD','EP_NOVEH','EP_GROUPQ']
    # variables = ['EP_POV150','EP_UNEMP','EP_NOHSDP','EP_UNINSUR','EP_AGE65','EP_AGE17','EP_DISABL','EP_SNGPNT','EP_LIMENG','EP_MINRTY','EP_MUNIT','EP_MOBILE','EP_CROWD','EP_NOVEH','EP_GROUPQ']

    BASE_DIR = '/home/h6x/git_projects/ornl-svi-data-processing/experiment_2/processed_data_1'
    DATA_DIR = f"{BASE_DIR}/npy_all_variables"
    COMBINED_FEATURES_DIR = f"{BASE_DIR}/npy_combined"

    # ANNOTATION_PATH = '/home/h6x/git_projects/data_processing/processed_data/svi_with_hepvu/2018/annotations_2018/annotation_NOD.csv'

    # output annotation path
    # OUTPUT_ANNOTATION_PATH = f'{BASE_DIR}/annotations_npy_2_classes_01.csv'


    file_names = {}

    for var in variables:

        var_dir = os.path.join(DATA_DIR, var)
        npy_filenames = get_npy_filenames(var_dir)

        file_names[var] = npy_filenames

    common_file_set = common_files(file_names)
    fips_codes = list(common_file_set)

    # Process each FIPS code
    for fips_code in tqdm(fips_codes, desc='Processing FIPS Codes'):
        print(f'Processing {fips_code}')
        
        # List to store persistence images for each variable
        persistence_images = []

        # Load the data for each variable
        for variable in variables:
            file_path = f'{DATA_DIR}/{variable}/{fips_code}.npy'
            
            # Check if the file exists before loading
            if os.path.exists(file_path):
                persistence_image = np.load(file_path)
                print(persistence_image.shape)
                persistence_images.append(persistence_image)
            else:
                print(f'File not found: {file_path}')
        
        # Concatenate the persistence images along the last axis
        combined_matrix = np.stack(persistence_images, axis=-1)
        fips_code_str = str(fips_code)

        # Save the combined persistence image
        output_path = f'{COMBINED_FEATURES_DIR}/{fips_code_str}.npy'
        np.save(output_path, combined_matrix)
        # print(f'Saved: {output_path}')

    print('Done processing FIPS codes')



    # print('Processing annotations...')

    # # full annotation df
    # annotation_df = pd.read_csv(ANNOTATION_PATH,dtype={'STCNTY':str})

    # # Filter the annotation dataframe to only include the FIPS codes that have been processed
    # annotation_df_filtered = annotation_df[annotation_df['STCNTY'].isin(fips_codes)]

    # # in annotation_df_filtered, the column "percen_US" change the values 0,1,2, values to 0 and  then turn the values 3,4,5 to 1
    # annotation_df_filtered['percen_US'] = annotation_df_filtered['percen_US'].replace({0: 0, 1: 0, 2: 0,3: 0})
    # annotation_df_filtered['percen_US'] = annotation_df_filtered['percen_US'].replace({4: 1})

    # # Save the filtered annotation dataframe
    # annotation_df_filtered.to_csv(OUTPUT_ANNOTATION_PATH, index=False)