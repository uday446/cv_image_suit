
import numpy as np
from tensorflow.keras.models import load_model
from cv_image_suit.utils import data_manager
from cv_image_suit.utils.config import config

class tfpredict:
    def __init__(self,filename):
        self.filename =filename
        self.confige = config("configs.json")
        self.dm = data_manager.datamanage()

    def predictiontf(self):
        self.param = self.confige.load_data()
        self.config_model = self.confige.configureModel(self.param)
        config_model = self.config_model

        # load model
        model_path = f"New_trained_model/{'new' + config_model['MODEL_NAME'] + '.h5'}"
        print('Loading...', model_path)
        model = load_model(model_path)

        # summarize model
        #model.summary()
        imagename = self.filename
        predict = self.dm.manage_input_data(imagename)
        result = model.predict(predict/255)
        results = np.argmax(result)
        names = self.dm.class_name()
        class_names = list(names.keys())
        print(class_names[results])

        return [{ "image_class" : str(class_names[results])}]



