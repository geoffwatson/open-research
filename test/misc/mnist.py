import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback
from components.models.models import MonitorableModel

import numpy as np


(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True)


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


def sample_n(test_set, n):
    inputs = []
    labels = []
    for ds_input, ds_label in test_set:
        for i in range(len(ds_input)):
            if len(inputs) < n:
                inputs.append(ds_input[i].numpy())
                labels.append(ds_label[i].numpy())
            else:
                break
    return np.array(inputs), np.array(labels)


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

input = tf.keras.layers.Input((28, 28, 1))
x1 = tf.keras.layers.Conv2D(2, strides=1, kernel_size=(3, 3), activation="relu", name='test-layer-1', padding='same', kernel_regularizer=l2(0.01))(input)
x2 = tf.keras.layers.Conv2D(16, strides=2, kernel_size=(3, 3), activation="relu", name='test-layer-2', padding='same', kernel_regularizer=l2(0.01))(x1)

x = tf.keras.layers.Flatten()(x2)
x = tf.keras.layers.Dense(10, activation='softmax')(x)

model = MonitorableModel(inputs=[input], outputs=[x], monitored_layers=[x1, x2])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
model.summary()

epochs = 6

sample_set = sample_n(ds_test, 5)

log_dir = '/Users/geoffwatson/workspace/tensorboard/mnist'
write = LambdaCallback(
    on_epoch_begin=lambda epoch, log: model.write_monitored_layers(log_dir, sample_set, step=epoch)
)

tensorboard = TensorBoard(log_dir=log_dir, write_images=True, write_graph=True)

model.fit(ds_train, epochs=epochs, validation_data=ds_test, callbacks=[tensorboard, write], verbose=1)
