from cv_image_suit.utils import exp_models_config
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

    def configureData(self,params):
        SIZE = params['IMAGE_SIZE'].split(',')
        BATCH = params['BATCH_SIZE'].split(',')
        BATCH = int(BATCH[0])
        h = int(SIZE[0])
        w = int(SIZE[0])
        temp = len(SIZE)-1
        c = int(SIZE[temp])
        IMAGE_SIZE = h, w, c
        CONFIG = {
            'TRAIN_DATA_DIR' : params['TRAIN_DATA_DIR'],
            'VALID_DATA_DIR' : params['VALID_DATA_DIR'],
            'AUGMENTATION': params['AUGMENTATION'],
            'CLASSES' : params['CLASSES'],
            'IMAGE_SIZE' : IMAGE_SIZE,
            'BATCH_SIZE' : BATCH,
            'PERCENT_DATA' : params['PERCENT_DATA']
        }

        return CONFIG


    def configureModel(self, params):
        self.mc = exp_models_config.modellconfig(params)
        opt = params['OPTIMIZER']
        CONFIG = {
            'MODEL_OBJ' : self.mc.return_model(),
            'EXP_NAME' : params['EXP_NAME'],
            'EPOCHS' : params['EPOCHS'],
            'OPTIMIZER' : opt[0],
            'FREEZE_ALL' : params['FREEZE_ALL'],
            'FREEZE_TILL' : params['FREEZE_TILL']
        }

        return CONFIG