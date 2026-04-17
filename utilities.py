from colorama import Fore
from glob import glob
import numpy as np
import os
import re

#---------------------------------------------------------------------------------

def checkpoint(DEBUG, msg="", col=""):
    if DEBUG:
        if msg != "":
            final_msg = Fore.GREEN + "Checkpoint: " + Fore.RESET
            final_msg += msg
        else:
            final_msg = Fore.Green + "Checkpoint!" + Fore.RESET
        
        final_msg += "\n"
        print(final_msg)

def error_message(condition, msg=""):
    if condition:
        if msg != "":
            error = Fore.RED + f"Error: {msg}"
        else:
            error = Fore.RED + "Error!"
        print(error)


#---------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def save_results(results, dir_name, gamma_0, h_0, N, M, idx=0):
    dir_path = os.path.join(BASE_DIR, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    file_name = f"{dir_path}/M{M}_N{N}_gamma{gamma_0}_h{h_0}_{idx}"

    try:
        np.save(file_name, results)
        print(f"File saved with name {file_name}")

    except Exception as e:
        print(f"Error during saving: {e}")


def read_results(dir_name, gamma_0=None, h_0=None, N=2048, M=None, idx=None):
    '''
        Returns a numpy matrix/array whose columns/values are the content
        of each file in dir_name. 

        NOTE: the number of columns/values in the output is fixed to N. 
        To perform analyses at different N (chain lengths), use read_final_values.
    '''
    dir_path = os.path.join(BASE_DIR, dir_name)
    file_name = ""
    if None not in [gamma_0, h_0, N, M, idx]:
        # If all parameters are provided, compose the name of the file
        file_name = f"M{M}_N{N}_gamma{gamma_0}_h{h_0}_{idx}"
    else:
        # Otherwise look all the files
        file_name = "*"

    query = os.path.join(dir_path, file_name)
    files = glob(query) 
    print(f"Reading from {query}")

    if not files:
        print("No files found!")
        return None

    size = (N, len(files)) if len(files) > 1 else N
    results = np.zeros(shape = size)
    for i, file in enumerate(files):
        try:
            if len(files) > 1:
                results[:, i] = np.load(file)
            else:
                results = np.load(file)
        except Exception as e:
            print(f"Error during reading: {e}")
    
    if None not in [gamma_0, h_0, N, M, idx]:
        return results
    else:
        # Extract the list of gamma-values in the same order as the files
        gamma_pattern = r"gamma(.*?)\_" 
        h_pattern     = r"_h(.*?)\_"
        
        gamma_list = []
        h_list     = []

        for file in files:
            gamma_match = re.search(gamma_pattern, file)
            gamma_list.append(gamma_match.group(1))

            h_match = re.search(h_pattern, file)
            h_list.append(h_match.group(1))

        return results, np.array(gamma_list, dtype=float), np.array(h_list, dtype=float)
    

def read_final_values(dir_name, save_gamma = False):
    '''
        Returns a dictionary with following characteristics:
            - if save_gamma is True, keys are (N,h0)-pairs and values are matrices with excitations/mm and gamma0 as rows

            - if save_gamma is False, keys are chain lengths (N) and values are matrices with excitations/mm and h0 as rows  
    '''

    # Construct absolute path
    dir_path = os.path.join(BASE_DIR, dir_name)
    file_name = "*"

    query = os.path.join(dir_path, file_name)
    files = glob(query) 
    print(f"Reading from {query}")

    if not files:
        print("No files found!")
        return None
    
    # Output dictionary
    results = {}

    N_pattern = r"_N(.*?)\_"
    h_pattern = r"_h(.*?)\_"
    if save_gamma:
        gamma_pattern = r"_gamma(.*?)\_"

    keys_list = [] # to keep track of new dictionary entries

    for file in files:
        # From each file extract N, h0, gamma0, and last value (excitation or magnetic moment)
        N_match = re.search(N_pattern, file)
        N_new = int(N_match.group(1))

        h_match = re.search(h_pattern, file)
        h_new = float(h_match.group(1))

        if save_gamma:
            gamma_match = re.search(gamma_pattern, file)
            gamma_new = float(gamma_match.group(1))

        loaded_vals = np.load(file)
        if loaded_vals.ndim > 0:
            new_val = loaded_vals[-2]
        else:
            new_val = loaded_vals

        # Append result to the dictionary
        if save_gamma:
            key_new = (N_new, h_new)
            if key_new not in keys_list:
                # If key_new is not yet a key in results, create a new entry in the dicttionary
                keys_list.append(key_new)
                results[key_new] = np.matrix([gamma_new, new_val], dtype=float)
            else:
                # If key_new is already a key, update the matrix by adding a new row
                new_row = np.matrix([gamma_new, new_val], dtype=float)
                results[key_new] = np.r_[results[key_new], new_row]

        else:
            if N_new not in keys_list:
                # If N_new is not yet a key in results, create a new entry in the dicttionary
                keys_list.append(N_new)
                results[N_new] = np.matrix([h_new, new_val], dtype=float)
            else:
                # If N_new is already a key, update the matrix by adding a new row
                new_row = np.matrix([h_new, new_val], dtype=float)
                results[N_new] = np.r_[results[N_new], new_row]
    
    sorted_res = {key: results[key] for key in sorted(results)}
    print(sorted_res.keys())
    return sorted_res

        
