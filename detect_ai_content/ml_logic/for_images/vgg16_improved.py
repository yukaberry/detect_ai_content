
from tensorflow.keras import layers, Model
from keras.applications.vgg16 import VGG16
from colorama import Fore, Style
import tensorflow as tf

from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

def clean_img_vgg16(user_input):

    """
    - cleaning (reshaping) image size that a user inputs for vgg16.
    - return cleaned img.

    """
    img = Image.open(BytesIO(user_input))
    print(f'img_to_array {img_to_array(img).shape}')

    # Resize to 224 x 224
    img = img.resize((224, 224))
    print(f'img_to_array {img_to_array(img).shape}')

    # Convert the image pixels to a numpy array
    arr = img_to_array(img)
    print(f'arr {arr.shape}')

    # 200704 ?
    # Reshape data for the model
    arr = arr.reshape((1, 224, 224, 3))
    print(f'arr {arr.shape}')

    # Prepare the image for the VGG model
    arr = preprocess_input(arr)
    print(f'arr {arr.shape}')

    return arr


def load_model():
    """
    - Return a keras VGG16 model (Baseline model, to re-train the model with bigger datasets)
    - Return None (but do not Raise) if no model is found
    """

    model = None

    print(Fore.BLUE + f"\nLoad, add layers and compile VGG16 model from local registry... (image MVP stlye)" + Style.RESET_ALL)

    base_model = VGG16(weights="imagenet",
                       include_top=False,
                       input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the VGG16 layers
    print(Fore.BLUE + f"\nbaseline done! " + Style.RESET_ALL)
    # Add custom layers on top
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    # activation=sigmoid : Single output for binary classification
    # human vs AI class is 1
    num_classes = 1
    x = layers.Dense(num_classes, activation='sigmoid')(x)
    print(Fore.BLUE + f"\nLAYERS done! " + Style.RESET_ALL)
    model = Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(Fore.BLUE + f"\COMPILE done! " + Style.RESET_ALL)
    return model
