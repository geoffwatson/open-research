import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np


class MonitorableModel(Model):
    '''
        Model that acts as an ordinary model, but allows passing in layers that can be printed out to image.
        Connect the functional model's layers outside this model, then pass in the inputs, outputs, and which layers
        should be written to image.

        To write to image, pass in a numpy-tuple of (inputs, labels) to #write_monitored_layers
    '''

    def __init__(self, inputs, outputs, monitored_layers=[]):
        super(MonitorableModel, self).__init__()
        if not isinstance(monitored_layers, list):
            monitored_layers = [monitored_layers]
        self.monitored_layers = monitored_layers
        self.monitored_layer_names = [l.name.split('/')[0] for l in self.monitored_layers]
        if not isinstance(outputs, list):
            outputs = [outputs]
        self.model = Model(inputs=inputs, outputs=outputs)
        self.monitorModel = Model(inputs=inputs, outputs=monitored_layers)
        self.output_size = len(outputs)

    def call(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)

    def summary(self, **kwargs):
        return self.model.summary(**kwargs)

    def write_monitored_layers(self, logdir, inputs, step):
        file_writer = tf.summary.create_file_writer(logdir)
        with file_writer.as_default():
            input_img = inputs[0]
            input_label = inputs[1]
            predictions = self.monitorModel(input_img)
            for layer_id in range(len(predictions)):
                for sample_id in range(len(predictions[layer_id])):
                    if sample_id < 2:
                        label = input_label[sample_id]

                        layer = predictions[layer_id][sample_id]
                        layer = np.moveaxis(layer, -1, 0)
                        layer = np.expand_dims(layer, -1)
                        for filter_id in range(len(layer)):
                            tf.summary.image(
                                ('%s/%s/%s' % (self.monitored_layer_names[layer_id], label, filter_id)),
                                np.expand_dims(layer[filter_id], 0),
                                step=step)
                    else:
                        break
