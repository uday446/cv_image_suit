import os
import time
import tensorflow as tf
from cv_image_suit.utils.experiment_utils.exp_config import config

class callback:
    """Configures callbacks for use in various training loops.
      Args:
          get_log_path: unique log location.
          checkpoint : None
      Returns:
          Instance of CallbackList used to control all Callbacks.
     """
    def __init__(self):
        self.config = config("Config_Layer/experiment_inputs_configs.json")
        param = self.config.load_data()
        final_data = self.config.configureModel(param)
        self.log_dir="Tensorboard/logs/"+final_data['EXP_NAME']

    def get_log_path(self):
      fileName = time.strftime("log_%Y_%m_%d_%H_%M_%S")
      logs_path = os.path.join(self.log_dir, fileName)
      print(f"Saving logs at {logs_path}")
      return logs_path


    def checkpoint(self):
        CKPT_path = "exp_Checkpoint/Model_ckpt.h5"
        checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)
        return  checkpointing_cb

