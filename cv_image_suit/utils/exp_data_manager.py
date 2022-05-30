from cv_image_suit.utils.exp_config import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from os import listdir

class datamanage:
    def __init__(self):
        self.confige = config("experiment_inputs_configs.json")

    def train_valid_generator(self):

        '''
        This function generates train & valid data from images path,
        it also performs data augmentation.
        :return object of data:
        '''
        class_mode = ""
        self.param = self.confige.load_data()
        self.config_data = self.confige.configureData(self.param)
        percentage = 100.0 - float(self.config_data['PERCENT_DATA'])
        percent = float(percentage/100)
        if self.config_data['CLASSES'] > 2:
            class_mode = "sparse"
        else:
            class_mode = "binary"
        if self.config_data['AUGMENTATION'] == 'True':
            print("Augmetation applied!")
            train_datagen = ImageDataGenerator(rescale=1. / 255,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True,
                                               validation_split=percent)

            valid_datagen = ImageDataGenerator(rescale=1. / 255,
                                               validation_split=percent)

            training_set = train_datagen.flow_from_directory(
                directory= self.config_data['TRAIN_DATA_DIR'],
                target_size=self.config_data['IMAGE_SIZE'][:-1],
                batch_size=self.config_data['BATCH_SIZE'],
                class_mode=class_mode,
                subset='training')

            valid_set = valid_datagen.flow_from_directory(
                directory=self.config_data['VALID_DATA_DIR'],
                target_size=self.config_data['IMAGE_SIZE'][:-1],
                batch_size=self.config_data['BATCH_SIZE'],
                class_mode=class_mode,
                subset='validation')

            return training_set, valid_set

        else:
            train_datagen = ImageDataGenerator(rescale=1. / 255,validation_split=percent)
            valid_datagen = ImageDataGenerator(rescale=1. / 255,validation_split=percent)

            training_set = train_datagen.flow_from_directory(
                directory=self.config_data['TRAIN_DATA_DIR'],
                target_size=self.config_data['IMAGE_SIZE'][:-1],
                batch_size=self.config_data['BATCH_SIZE'],
                class_mode=class_mode,
                subset='training')

            valid_set = valid_datagen.flow_from_directory(
                directory=self.config_data['VALID_DATA_DIR'],
                target_size=self.config_data['IMAGE_SIZE'][:-1],
                batch_size=self.config_data['BATCH_SIZE'],
                class_mode=class_mode,
                subset='validation')

            return training_set, valid_set

    #def data_percent(self,train,valid,clas,percent):
        #if os.path.exists("temp/"):
            #os.removedirs()





