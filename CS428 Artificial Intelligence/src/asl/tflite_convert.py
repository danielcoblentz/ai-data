"""Convert a Keras model to TFLite"""
import tensorflow as tf
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to input Keras model (.h5)")
ap.add_argument("-o", "--output", required=True,
                help="path to output TFLite file")
args = vars(ap.parse_args())

# load the Keras model
model = tf.keras.models.load_model(args['model'])

# convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# write the model to disk
open(args['output'], 'wb').write(tflite_model)
