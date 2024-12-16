import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from src.cnnClassifier.config.configuration import PrepareBaseModelConfig
from pathlib import Path

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        # 1. Layer Freezing (Transfer Learning):
        """Controls which layers can be trained

        freeze_all: If True, makes all layers non-trainable
        freeze_till: Freezes layers up to a specific point"""
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        # 2. Adding classification layers
        flatten_in = tf.keras.layers.Flatten()(model.output) #Converts the 2D feature maps to 1D vector
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in) #Adds final classification layer

        # 3. Creating full model - Combines base model with new classification head

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # 4. Compiling model
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),   #Uses SGD (Stochastic Gradient Descent) optimizer
            loss=tf.keras.losses.CategoricalCrossentropy(),   #Categorical Cross-entropy loss for multi-class classification
            metrics=["accuracy"] #Tracks accuracy during training
        )

        full_model.summary()
        return full_model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    