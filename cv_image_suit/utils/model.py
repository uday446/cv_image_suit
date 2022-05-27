from cv_image_suit.utils.config import config
import tensorflow as tf
from tensorflow.keras import models

class modell:
    def __init__(self):
        self.confige = config("configs.json")

    def get_model(self):
        """The logic for loading pretrain model.

         Args:
          MODEL_OBJ: Model object

        Returns:
          It returns keras model objcet

        """
        try:
            self.param = self.confige.load_data()
            self.config_model = self.confige.configureModel(self.param)
            self.config_data = self.confige.configureData(self.param)
            model =  self.config_model['MODEL_OBJ']
            print("Detected pretrain model!!")
            return model

        except Exception as e:
            print("Something went wrong!!", e)


    def model_preparation(self,model):
        print('Preparing model...')
        if self.config_model['FREEZE_ALL'] == 'True':
            print('Freezing all...')
            for layer in model.layers:
                layer.trainable = False

            # Add custom layers
            flatten_layer = tf.keras.layers.Flatten()

            if self.config_data['CLASSES'] > 2:
                print('Adding softmax...')
                prediction = tf.keras.layers.Dense(
                    units=self.config_data['CLASSES'],
                    activation="softmax"
                )#(flatten_in)

                full_model = models.Sequential([
                                model,
                                flatten_layer,
                                prediction
                            ])

                full_model.compile(
                    optimizer=self.config_model['OPTIMIZER'],
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"]
                )
                print('Model loaded!!')

                return full_model

            else:
                print('Adding sigmoid...')
                prediction = tf.keras.layers.Dense(
                    units=1,
                    activation="sigmoid"
                )#(flatten_in)

                full_model = models.Sequential([
                                model,
                                flatten_layer,
                                prediction
                            ])

                full_model.compile(
                    optimizer=self.config_model['OPTIMIZER'],
                    loss="binary_crossentropy",
                    metrics=["accuracy"]
                )

                print('Model loaded!!')

                return full_model


        else:

            for layer in model.layers[:self.config_model['FREEZE_TILL']]:
                layer.trainable = False

                # Add custom layers
                flatten_layer = tf.keras.layers.Flatten()#(model.output)

                if self.config_data['CLASSES'] > 2:
                    prediction = tf.keras.layers.Dense(
                        units=self.config_data['CLASSES'],
                        activation="softmax"
                    )#(flatten_in)

                    full_model = models.Sequential([
                        model,
                        flatten_layer,
                        prediction
                    ])

                    full_model.compile(
                        optimizer=self.config_model['OPTIMIZER'],
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"]
                    )
                    print('Model loaded!!')

                    return full_model

                else:
                    prediction = tf.keras.layers.Dense(
                        units=1,
                        activation="sigmoid"
                    )#(flatten_in)

                    full_model = models.Sequential([
                        model,
                        flatten_layer,
                        prediction
                    ])

                    full_model.compile(
                        optimizer=self.config_model['OPTIMIZER'],
                        loss="binary_crossentropy",
                        metrics=["accuracy"]
                    )

                    print('Model loaded!!')

                    return full_model

    def load_pretrain_model(self):

        """The logic for loading pretrain model.

         Args:
          MODEL_OBJ: Model object

        Returns:
          It returns keras model objcet

        """
        model = self.get_model()
        model = self.model_preparation(model)
        return model


    def load_exist_model(self):

        """The logic for loading an existing model.

         Args:
          PRETRAIN_MODEL_DIR: Your existing model path

        Returns:
          It returns keras model objcet

        """
        print('Loading existing model...')
        print("Model loaded!")
        model = tf.keras.models.load_model(self.config_model['PRETRAIN_MODEL_DIR'])
        model = self.model_preparation(model)
        return model










