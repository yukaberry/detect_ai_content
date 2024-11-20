

# from tensorflow.python.keras.layers import Input, Dense
# from tensorflow.python.keras.models import Sequential, Layer
# import tensorflow.python.keras.layers as layers
# from keras._tf_keras.keras.preprocessing import image_dataset_from_directory

from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.models as keras_models

from keras.layers import Dense, Input
import keras.layers as layers

from keras import Sequential
from keras.models import Sequential
from keras._tf_keras.keras.preprocessing import image_dataset_from_directory

import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image

import numpy as np

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

class TrueNetImageResNet50:
    def _load_model(self):
        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        model_path = os.path.join(f'{module_dir_path}/..', 'detect_ai_content', 'models', 'achmed', 'achmed_ResNet50.keras')
        model = keras_models.load_model(model_path)
        return model

    def __init__(self):
        self.name = "TrueNetImageResNet50"
        self.description = ""
        self.mlflow_model_name = "TrueNetImageResNet50"
        self.mlflow_experiment = "TrueNetImageResNet50_experiment_leverdewagon"
        self.model = self._load_model()

    def get_architecture_model():
        # Load pre-trained ResNet50 model without the top layer
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

        # Freeze the base model
        base_model.trainable = False  # Freeze the VGG16 layers
        for layer in base_model.layers:
            layer.trainable = False

        # Add custom layers for binary classification
        x = Flatten()(base_model.output)
        x = Dense(16, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification

        # Define the final model
        model = Model(inputs=base_model.input, outputs=output)
        return model

    def compile_model(model):
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def retrain_full_model():
        # get images
        # train
        # evaluate
        model = TrueNetImageResNet50.get_architecture_model()
        model = TrueNetImageResNet50.compile_model(model)
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
            image_size=(128, 128),
            batch_size=batch_size,
            subset = "both")

        training_set = images_dataset[0]
        validation_set = images_dataset[1]

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        model_path = os.path.join(f'{module_dir_path}/..', 'detect_ai_content', 'models', 'achmed', 'achmed_ResNet50.keras')
        model.save(model_path)

        model.fit(
            training_set,
            epochs = 1,
            validation_data = validation_set,
            callbacks=[early_stopping]
        )

        # print(model)

        import detect_ai_content
        module_dir_path = os.path.dirname(detect_ai_content.__file__)
        model_path = os.path.join(f'{module_dir_path}/..', 'detect_ai_content', 'models', 'achmed', 'achmed_ResNet50.keras')
        model.save(model_path)

    def pre_process(img):

        # Resize to 128, 128
        img = img.resize((128, 128))
        img_array = image.img_to_array(img)
        img_array = img_array/255
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, user_input):

        """
        Return
        1. prediciton class '1' or '0'
        2. prediction message 'AI' or ' Human'

        """
        # preprocess img
        to_predict_img_array = TrueNetImageResNet50.pre_process(user_input)

        # predict probablity
        # output is a probability value (between 0 and 1)
        prediction_results = self.model.predict(to_predict_img_array)
        class_prediction = prediction_results[0][0]
        print(prediction_results)
        predict_prob_confidence = abs(class_prediction - 0.5)/0.5

        # Get predicted indices
        # do not use np.argmax for binary, it is for multi-class!!
        # predicted_probabilities = np.argmax(predicted_class, axis=1)

        # predict class
        predicted_class = int(class_prediction > 0.5)

        # 0 likely representing 'FAKE' and 1 representing 'REAL'
        if predicted_class == 1:
            prediction_message = "Predicted as AI"
        elif predicted_class == 0:
            prediction_message = "Predicted as Human"

        return predicted_class, prediction_message, predict_prob_confidence
