import sys

import numpy as np
from tensorflow.keras.models import load_model
from cv_image_suit.utils.normal_utils import data_manager
from cv_image_suit.utils.normal_utils.config import config
from cv_image_suit.Exception_Layer.exception import GenericException
from cv_image_suit.Logging_Layer.logger import App_Logger
class tfpredict:
    def __init__(self,filename):
        self.filename =filename
        self.confige = config("Config_Layer/configs.json")
        self.dm = data_manager.datamanage()
        self.logger = App_Logger()

    def predictiontf(self):
        try:
            self.param = self.confige.load_data()
            self.config_model = self.confige.load_pred_data()
            self.data_config = self.confige.configureData(self.param)

            # load model
            model_path = f"New_trained_model/{self.config_model['MODEL_NAME']}"
            print('Loading...', model_path)
            model = load_model(model_path)

            # summarize model
            #model.summary()
            imagename = self.filename
            predict = self.dm.manage_input_data(imagename)
            names = self.dm.class_name()
            class_names = list(names.keys())
            if self.data_config['CLASSES'] > 2:
                result = model.predict(predict/255)
                results = np.argmax(result)
                return [{ "image_class" : str(class_names[results])}]
            else:
                result = model.predict(predict/255)
                predicted_classes = [1 * (x[0] >= 0.5) for x in result]
                return [{ "image_class" : str(class_names[predicted_classes[0]])}]
        except Exception as e:
            exception = GenericException(
                "Failed during training in module [{0}] class [{1}] method[{2}]"
                .format(self.__module__, tfpredict.__name__,
                         self.predictiontf.__name__))
            file = open("Logging_Layer/Logs/logs.txt", 'a+')
            exception_msg = exception.error_message_detail(str(e), sys)
            self.logger.log(file, "Exception occured %s " % exception_msg)
            file.close()
            raise Exception(exception_msg)





