from tensorflow.keras.callbacks import LambdaCallback


class AverageActivationImageCallback(LambdaCallback):

    def __init__(self, layer):
        super(self, AverageActivationImageCallback).__init__(
            on_epoch_begin=self.log_epoch_begin,
            on_train_end=self.log_train_end)
        self.layer = layer

    def log_epoch_begin(self, epoch, log):
        pass

    def log_train_end(self, log):
        pass