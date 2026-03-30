from colorama import Fore
import numpy as np
import os

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

def save_decimations(decimation_list, gamma_0, h_0, N, M, idx=0, dir="saves"):
    os.makedirs(dir, exist_ok=True)
    file_name = f"{dir}/decimations_M{M}_N{N}_gamma{gamma_0}_h{h_0}_{idx}"

    try:
        np.save(file_name, decimation_list)
        print(f"File saved with name {file_name}")
        
    except Exception as e:
        print(f"Error during saving: {e}")
