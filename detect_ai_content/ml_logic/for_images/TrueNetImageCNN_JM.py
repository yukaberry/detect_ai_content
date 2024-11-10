

# from tensorflow.python.keras.layers import Input, Dense
# from tensorflow.python.keras.models import Sequential, Layer
# import tensorflow.python.keras.layers as layers
# from keras._tf_keras.keras.preprocessing import image_dataset_from_directory

from tensorflow import keras

from keras.layers import Dense, Input
import keras.layers as layers

from keras import Sequential
from keras.models import Sequential
from keras._tf_keras.keras.preprocessing import image_dataset_from_directory

import os

from detect_ai_content.ml_logic.mlflow import mlflow_save_metrics, mlflow_save_model, mlflow_save_params, load_model


## CHECK GPU Capabiliteis
import sys

import tensorflow.keras
import pandas as pd
import sklearn as sk
import scipy as sp
import tensorflow as tf
import platform

print(f"Python Platform: {platform.platform()}")
print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print(f"SciPy {sp.__version__}")
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

class TrueNetImageCNN_JM:
    def _load_model(self, stage="Production"):
        """
        Model sumary :
            Trained Found 120000 files belonging to 2 classes. Using 84000 files for training. Using 36000 files for validation.
            Algo : light_model
            Cross Validate average result (0.2 test) : accuracy: 0.8275
        """
        return load_model(self.mlflow_model_name, is_tensorflow=False, stage=stage)

    def __init__(self):
        self.name = "TrueNetImageCNN_JM"
        self.description = ""
        self.mlflow_model_name = "TrueNetImageCNN_JM"
        self.mlflow_experiment = "TrueNetImageCNN_JM_experiment_leverdewagon"
        self.model = self._load_model()

    def get_architecture_light_model():
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(64, 64, 3)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))
        return model

    def get_architecture_model():
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(128, 128, 3)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))
        return model

    def compile_model(model):
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def retrain_full_model():
        # get images
        # train
        # evaluate
        model = TrueNetImageCNN_JM.get_architecture_light_model()
        model = TrueNetImageCNN_JM.compile_model(model)
        print(model.summary())
        # history = model.fit(train_images, train_labels, epochs=10,
        #             validation_data=(test_images, test_labels))

        batch_size = 64

        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)

        images_dataset = image_dataset_from_directory(
            f"{module_dir_path}/../raw_data/kaggle-cifake-real-and-ai-generated-synthetic-images",
            labels = "inferred", # inferred from sub folder name
            label_mode = "binary",
            validation_split=0.3,
            seed=123,
            image_size=(64, 64),
            batch_size=batch_size,
            subset = "both")

        training_set = images_dataset[0]
        validation_set = images_dataset[1]

        fitting_model = model.fit(
            training_set,
            epochs = 1,
            validation_data = validation_set
        )

        print(fitting_model)


from keras.applications.vgg16 import VGG16
class TrueNetImageCNN_vgg16_JM:
    def _load_model(self, stage="Production"):
        """
        Model sumary :
            Trained Found 120000 files belonging to 2 classes. Using 84000 files for training. Using 36000 files for validation.
            Algo : light_model
            Cross Validate average result (0.2 test) : accuracy: 0.8221
        """
        return load_model(self.mlflow_model_name, is_tensorflow=False, stage=stage)

    def __init__(self):
        self.description = ""
        self.name = "TrueNetImageCNN_vgg16_JM"

    def get_architecture_light_model():
        model = Sequential()
        model.add(layers.InputLayer(input_shape=(64, 64, 3)))

        VGG16_model = VGG16(weights="imagenet",
                            include_top=False,
                            input_shape=(64, 64, 3))
        VGG16_model.trainable = False  # Freeze the VGG16 layers
        model.add(VGG16_model)
        model.add(layers.Flatten())
        model.add(layers.Dense(16, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))
        return model

    def compile_model(model):
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def retrain_full_model():
        # get images
        # train
        # evaluate
        model = TrueNetImageCNN_vgg16_JM.get_architecture_light_model()
        model = TrueNetImageCNN_vgg16_JM.compile_model(model)
        print(model.summary())
        # history = model.fit(train_images, train_labels, epochs=10,
        #             validation_data=(test_images, test_labels))

        batch_size = 64

        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)

        images_dataset = image_dataset_from_directory(
            f"{module_dir_path}/../raw_data/kaggle-cifake-real-and-ai-generated-synthetic-images",
            labels = "inferred", # inferred from sub folder name
            label_mode = "binary",
            validation_split=0.3,
            seed=123,
            image_size=(64, 64),
            batch_size=batch_size,
            subset = "both")

        training_set = images_dataset[0]
        validation_set = images_dataset[1]

        fitting_model = model.fit(
            training_set,
            epochs = 1,
            validation_data = validation_set
        )

        print(fitting_model)
