import tensorflow as tf
import pandas as pd
import os

from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model

import pandas as pd
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input



from colorama import Fore, Style


def load_model():
    """
    - Return a keras VGG16 model (Baseline model, to re-train the model with bigger datasets)
    - Return None (but do not Raise) if no model is found
    """

    model = None

    print(Fore.BLUE + f"\nLoad CNN baseline model from local registry..." + Style.RESET_ALL)

    base_model = load_model('../models/cnn_model.h5')
    # preventing model from being updated during training.
    base_model.trainable = False

    print(Fore.BLUE + f"\nbaseline done! " + Style.RESET_ALL)

    return model


def clean_img(user_input):

    img = Image.open(BytesIO(user_input))


    img = tf.keras.utils.image_dataset_from_directory(
                        img,
                        labels='inferred',
                        validation_split=0.2,
                        subset="validation",
                        seed=123,
                        image_size=(180, 180),
                        batch_size=64
                        )

    # Resize to 224 x 224
    img = img.resize((224, 224))

    # Convert the image pixels to a numpy array
    img = img_to_array(img)

    # Reshape data for the model
    img = img.reshape((1, 224, 224, 3))

    # Prepare the image for the VGG model
    img = preprocess_input(img)
