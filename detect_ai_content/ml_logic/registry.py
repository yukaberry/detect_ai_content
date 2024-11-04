import glob
import os
import pickle
import detect_ai_content

from colorama import Fore, Style

def load_model():
    """
    Return a saved model:
    - locally (latest one in alphabetical order)

    Return None (but do not Raise) if no model is found
    """

    latest_model = None

    print(Fore.BLUE + f"\nLoad repo-saved model from local registry... (MVP stlye)" + Style.RESET_ALL)

    # Get the hardcoded model
    module_path = os.path.abspath(detect_ai_content.__file__)
    dir_path = os.path.dirname(os.path.realpath(module_path))
    local_one_model_path = os.path.join(dir_path, "models/leverdewagon/finalized_genai_text_detection_model.pickle")

    latest_model = pickle.load(open(local_one_model_path, 'rb'))

    print("✅ Model loaded from local disk")

    return latest_model
