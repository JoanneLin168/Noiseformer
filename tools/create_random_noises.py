import os
import numpy as np
import argparse

def get_noise_params():
    return {
        'alpha_brightness': np.round(np.random.uniform(0.05, 0.3), decimals=2),
        'gamma_brightness': np.round(np.random.uniform(0.1, 1), decimals=2),
        'shot_noise': np.round(np.random.uniform(0, 0.5), decimals=2),
        'read_noise': np.round(np.random.uniform(0, 0.1), decimals=2),
        'uniform_noise': np.round(np.random.uniform(0, 0.1), decimals=2),
        'row_noise': np.round(np.random.uniform(0, 0.01), decimals=3),
        'row_noise_temp': np.round(np.random.uniform(0, 0.01), decimals=3),
        'periodic0': np.round(np.random.uniform(0, 0.5), decimals=2),
        'periodic1': np.round(np.random.uniform(0, 0.5), decimals=2),
        'periodic2': np.round(np.random.uniform(0, 0.5), decimals=2),
    }

def process_folder(folder_path, csv_path, first_folder=False):
    noise_params = get_noise_params()
    
    # Write headers if this is the first folder
    if first_folder:
        with open(csv_path, 'w') as f:
            headers = ['folder'] + list(noise_params.keys())
            f.write(','.join(headers) + '\n')
    
    # Append values
    with open(csv_path, 'a') as f:
        folder_name = os.path.basename(folder_path)
        values = [folder_name] + [str(v.item()) for v in noise_params.values()]
        f.write(','.join(values) + '\n')

def main(input_path):
    csv_path = './noise_parameters.csv'
    
    # Iterate over each folder in the provided input_path
    folders = [item for item in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, item))]
    
    for i, folder in enumerate(folders):
        folder_path = os.path.join(input_path, folder)
        process_folder(folder_path, csv_path, first_folder=(i==0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate noise parameters for folders")
    parser.add_argument("-i", "--input", required=True, help="Path to the validation set folder")
    args = parser.parse_args()

    main(args.input)