from cv_image_suit.utils.config import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

class datamanage:
    def __init__(self):
        self.confige = config("configs.json")

    def train_valid_generator(self):

        '''
        This function generates train & valid data from images path,
        it also performs data augmentation.
        :return object of data:
        '''
        class_mode = ""
        self.param = self.confige.load_data()
        self.config_data = self.confige.configureData(self.param)
        if self.config_data['CLASSES'] > 2:
            class_mode = "sparse"
        else:
            class_mode = "binary"
        if self.config_data['AUGMENTATION'] == 'True':
            print("Augmetation applied!")
            train_datagen = ImageDataGenerator(rescale=1. / 255,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True)

            valid_datagen = ImageDataGenerator(rescale=1. / 255)

            training_set = train_datagen.flow_from_directory(
                directory= self.config_data['TRAIN_DATA_DIR'],
                target_size=self.config_data['IMAGE_SIZE'][:-1],
                batch_size=self.config_data['BATCH_SIZE'],
                class_mode=class_mode)

            valid_set = valid_datagen.flow_from_directory(
                directory=self.config_data['VALID_DATA_DIR'],
                target_size=self.config_data['IMAGE_SIZE'][:-1],
                batch_size=self.config_data['BATCH_SIZE'],
                class_mode=class_mode)

            return training_set, valid_set

        else:
            train_datagen = ImageDataGenerator(rescale=1. / 255)
            valid_datagen = ImageDataGenerator(rescale=1. / 255)

            training_set = train_datagen.flow_from_directory(
                directory=self.config_data['TRAIN_DATA_DIR'],
                target_size=self.config_data['IMAGE_SIZE'][:-1],
                batch_size=self.config_data['BATCH_SIZE'],
                class_mode=class_mode)

            valid_set = valid_datagen.flow_from_directory(
                directory=self.config_data['VALID_DATA_DIR'],
                target_size=self.config_data['IMAGE_SIZE'][:-1],
                batch_size=self.config_data['BATCH_SIZE'],
                class_mode=class_mode)

            return training_set, valid_set

    def class_name(self):
        train, valid = self.train_valid_generator()
        return train.class_indices

    def manage_input_data(self,INPUT_IMAGE):
        '''
        This function takes an new raw image & convert the image
        into nd array with respect to training images dimention.
        :param INPUT_IMAGE:
        :return nd array:
        '''
        try:
            self.param = self.confige.load_data()
            self.config_data = self.confige.configureData(self.param)
            INPUT_IMG = INPUT_IMAGE
            test_image = image.load_img(INPUT_IMG, target_size= self.config_data['IMAGE_SIZE'][:-1])
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)

            return test_image
        except Exception as e:
            print(e)