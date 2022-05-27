
from cv_image_suit.utils import model
from cv_image_suit.utils import data_manager
from cv_image_suit.utils.config import config
from cv_image_suit.utils import callbacks
import tensorflow as tf

class tftrainer:
    def __init__(self):
        self.confige = config("configs.json")
        self.call = callbacks.callback()
        self.mod = model.modell()
        self.dm = data_manager.datamanage()

    def train(self):

        """The logic for one training step.

        This method should contain the mathematical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.

         Args:
          data: A nested structure of `Tensor`s.

        Returns:
          A `dict` containing values.Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.

        """
        self.param = self.confige.load_data()
        self.config_model = self.confige.configureModel(self.param)
        self.config_data = self.confige.configureData(self.param)
        self.log_dir = self.call.get_log_path()
        self.ckp = self.call.checkpoint()
        model_obj = self.mod.load_pretrain_model()
        my_model = model_obj
        train_data, valid_data = self.dm.train_valid_generator()

        #callbacks
        log_dir = self.log_dir
        tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        ckp = self.ckp

        call = [tb_cb, ckp]

        #Calculating steps_per_epoch & validation_steps
        steps_per_epoch = train_data.samples // train_data.batch_size
        validation_steps = valid_data.samples // valid_data.batch_size

        my_model.fit(
            train_data,
            validation_data=valid_data,
            epochs=self.config_model['EPOCHS'],
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=call
        )

        new_path = f"New_trained_model/{'new'+self.config_model['MODEL_NAME']+'.h5'}"
        my_model.save(new_path)
        print(f"Model saved at the following location : {new_path}")
