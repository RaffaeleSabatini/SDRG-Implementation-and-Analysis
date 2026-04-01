from colorama import Fore
from glob import glob
import numpy as np
import os
import re

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


#------------------------------------------------------

def save_results(results, kind, gamma_0, h_0, N, M, idx=0):
    os.makedirs(kind, exist_ok=True)
    file_name = f"{kind}/M{M}_N{N}_gamma{gamma_0}_h{h_0}_{idx}"

    try:
        np.save(file_name, results)
        print(f"File saved with name {file_name}")

    except Exception as e:
        print(f"Error during saving: {e}")


def read_results(dir, gamma_0=None, h_0=None, N=2048, M=None, idx=None):
    file_name = ""
    if None not in [gamma_0, h_0, N, M, idx]:
        # If all parameters are provided, compose the name of the file
        file_name = f"M{M}_N{N}_gamma{gamma_0}_h{h_0}_{idx}"
    else:
        # Otherwise look all the files
        file_name = "*"

    query = f"{dir}/{file_name}"
    files = glob(query) 
    print(f"Reading from {query}")

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
        pattern = r"gamma(.*?)\_"
        gamma_list = []
        for file in files:
            match = re.search(pattern, file)
            gamma_list.append(match.group(1))

        return results, np.array(gamma_list)
