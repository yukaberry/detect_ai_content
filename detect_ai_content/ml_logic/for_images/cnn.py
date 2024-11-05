from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from colorama import Fore, Style
import os

import detect_ai_content


# TODO
# differenciate 'load_model()' until Class is made

def load_cnn_model():

    print(Fore.BLUE + f"\nLoad CNN model " + Style.RESET_ALL)

    # PATH
    # get absolute path to the Python file for detect_ai_content
    module_path = os.path.abspath(detect_ai_content.__file__)
    # extract the directory path containing this module
    dir_path = os.path.dirname(os.path.realpath(module_path))
    # build the path to the CNN model
    local_one_model_path = os.path.join(dir_path,
                                        "models/yukaberry/cnn_model.h5")
    print(local_one_model_path)
    # load model
    latest_model = load_model(local_one_model_path)

    print(Fore.BLUE + f"\n ✅ Model loaded " + Style.RESET_ALL)

    return latest_model


# TODO
# 1. differenciate 'clean_img()' until Class is made
# 2. reshape if image is JPEG (180, 180, 4)

def clean_img_cnn(img_path, target_size=(180, 180), color_mode='rgb'):

    """
    - input uer's image
    - change image shape
    - color_mode='rgb'(180,180,3)  or color_mode='rgba'(180,180,4)
      (1 for grayscale)

    """

    # Load the image
    # reshape size (180,180,3)
    img = image.load_img(img_path, target_size=target_size, color_mode=color_mode)

    # Convert img to np array
    img_arr = image.img_to_array(img)

    # one image, size of image, size of image, colour mode (1, 180, 180, 3)
    img_arr = np.expand_dims(img_arr, axis=0)

    # normalize [0, 1]
    img_arr /= 255.0

    return img_arr
