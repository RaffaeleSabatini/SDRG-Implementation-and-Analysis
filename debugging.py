from colorama import Fore

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