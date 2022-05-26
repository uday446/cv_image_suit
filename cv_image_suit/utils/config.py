from cv_image_suit.utils import models_config
import json

class config:
    """Public cv_image_suit utilities.
    This module is used as a shortcut to access all the symbols. Those symbols was
    exposed under train engine and predict engine.
    """
    def __init__(self,filename):
        self.filename =filename

    def load_data(self):
        with open(self.filename,'r') as f:
              params = json.load(f)
        return params

    #Data config
    #TRAIN_DATA_DIR = params['TRAIN_DATA_DIR']
    #print(TRAIN_DATA_DIR)
    #VALID_DATA_DIR = params['VALID_DATA_DIR']
    #CLASSES = params['CLASSES']
    #SIZE = params['IMAGE_SIZE'].split(',')
    #h = int(SIZE[0])
    #w = int(SIZE[1])
    #c = int(SIZE[2])
    #IMAGE_SIZE = h,w,c
    #AUGMENTATION = params['AUGMENTATION']
    #BATCH_SIZE = params['BATCH_SIZE']

    # Model config
    #MODEL_OBJ = params['MODEL_OBJ']
    #print("I am Model obj", MODEL_OBJ)
    #MODEL_OBJ = mc.return_model(MODEL_OBJ)
    #MODEL_NAME = params['MODEL_NAME']
    #EPOCHS = params['EPOCHS']
    #OPTIMIZER = params['OPTIMIZER']
    #LOSS_FUNC = params['LOSS_FUNC']
    #FREEZE_ALL = params['FREEZE_ALL']
    #return



    def configureData(self,params):
        SIZE = params['IMAGE_SIZE'].split(',')
        h = int(SIZE[0])
        w = int(SIZE[1])
        c = int(SIZE[2])
        IMAGE_SIZE = h, w, c
        CONFIG = {
            'TRAIN_DATA_DIR' : params['TRAIN_DATA_DIR'],
            'VALID_DATA_DIR' : params['VALID_DATA_DIR'],
            'AUGMENTATION': params['AUGMENTATION'],
            'CLASSES' : params['CLASSES'],
            'IMAGE_SIZE' : IMAGE_SIZE,
            'BATCH_SIZE' : params['BATCH_SIZE'],
        }

        return CONFIG


    def configureModel(self, params):
        self.pram = self.load_data()
        self.mc = models_config.modellconfig(self.pram)
        CONFIG = {
            'MODEL_OBJ' : self.mc.return_model(),
            'MODEL_NAME' : params['MODEL_NAME'],
            'EPOCHS' : params['EPOCHS'],
            'OPTIMIZER' : params['OPTIMIZER'],
            'FREEZE_ALL' : params['FREEZE_ALL']
        }

        return CONFIG