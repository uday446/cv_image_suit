import json
import os
import numpy
import tensorflow as tf
from cv_image_suit.utils.exp_config import config
from cv_image_suit.utils import exp_callbacks
from cv_image_suit.utils import exp_model
from cv_image_suit.utils import exp_data_manager

class modelfinder:
    def __init__(self,filename):
        self.filename = filename
        self.config = config(self.filename)
        self.call = exp_callbacks.callback()
        self.mod = exp_model.modell()
        self.dm = exp_data_manager.datamanage()

    def find_model(self):
        param = self.config.load_data()
        final_data = self.config.configureData(param)
        final_model_data = self.config.configureModel(param)

        self.log_dir = self.call.get_log_path()
        model_obj = self.mod.load_pretrain_model()
        my_model = model_obj
        #train_dir,valid_dir = self.dm.data_percent(final_data['TRAIN_DATA_DIR'],final_data['VALID_DATA_DIR'],final_data['CLASSES'],final_data['PERCENT_DATA'])
        train_data, valid_data = self.dm.train_valid_generator()

        #callbacks
        log_dir = self.log_dir
        tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        call = [tb_cb]

        #Calculating steps_per_epoch & validation_steps
        steps_per_epoch = train_data.samples // train_data.batch_size
        validation_steps = valid_data.samples // valid_data.batch_size
        acc = {}
        i = 0
        model_names = param['MODEL_OBJ']
        for x in my_model:
            history = x.fit(
                train_data,
                validation_data=valid_data,
                epochs=final_model_data['EPOCHS'],
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=call
            )
            new_path = f"EXP_New_trained_model/{'new'+model_names[i]+'.h5'}"
            x.save(new_path)
            print(f"Model saved at the following location : {new_path}")

            val_acc = numpy.max(history.history['val_accuracy'])
            acc[val_acc] = model_names[i]
            i = i + 1

        temp = []
        for res in acc.keys():
            temp.append(float(res))
        max_acc = numpy.max(temp)
        best_model = acc[max_acc]
        CONFIG = {"BEST_MODEL":best_model}

        with open("experiment_result.json","w") as f:
            json.dump(CONFIG,f)

        for i in temp:
            if os.path.exists("EXP_New_trained_model/"+acc[i]):
                if acc[i] != best_model:
                    os.remove("EXP_New_trained_model/"+acc[i])














