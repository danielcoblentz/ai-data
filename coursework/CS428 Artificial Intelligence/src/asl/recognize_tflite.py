import cv2
import numpy as np
import argparse
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    import tensorflow as tf
    tflite = tf.lite

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to TFLite model")
ap.add_argument("-l", "--labels", required=True,
                help="path to label binarizer")
args = vars(ap.parse_args())

# load tflite model
interpreter = tflite.Interpreter(model_path=args['model'])
interpreter.allocate_tensors()

# get input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def classify(image):
    # Preprocess to expected size
    inp = cv2.resize(image, (64,64))
    inp = inp.astype('float32')/255.0
    inp = np.expand_dims(inp, axis=0)
    if inp.ndim == 3:
        inp = np.expand_dims(inp, -1)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    return out
