from src.cnnClassifier.constants import *
from src.cnnClassifier.utils.common import read_yaml, create_directories
import tensorflow as tf
from src.cnnClassifier.config.configuration import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,   # Normalizes pixel values to [0,1]
            validation_split=0.20  # 20% of data used for validation
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Image dimensions
            batch_size=self.config.params_batch_size,    # Number of images per batch
            interpolation="bilinear"  # Image resizing method
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

            #"""Creates generator for validation data
            #Only applies rescaling, no augmentation
            #shuffle=False keeps validation data order consistent"""
        
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,           # Random rotations up to 40 degrees
                horizontal_flip=True,        # Randomly flip images horizontally
                width_shift_range=0.2,       # Shift width by up to 20%
                height_shift_range=0.2,      # Shift height by up to 20%
                shear_range=0.2,            # Apply shearing transformations
                zoom_range=0.2,             # Random zoom up to 20%
                **datagenerator_kwargs      # Include base configurations
            )
        else:
            train_datagenerator = valid_datagenerator  # No augmentation

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,   # Shuffle training data
            **dataflow_kwargs
        )

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)



    
    def train(self):
        # Calculate steps per epoch for training and validation
        """Calculates how many batches are needed to complete one epoch
        // operator performs integer division"""
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size



        # Train the model
        self.model.fit(
            self.train_generator,            # Source of training data - Provides batches of training data with augmentation
            epochs=self.config.params_epochs,# Number of complete passes through the data
            steps_per_epoch=self.steps_per_epoch,  # Batches per epoch - Number of batches in one epoch
            validation_steps=self.validation_steps, # Validation batches -  Number of validation batches
            validation_data=self.valid_generator   # Validation data source -Generator for validation data
        )
        
        # Save the trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )