import tensorflow as tf
import json

#load model structure
path = './Model/'
f = open(path + 'model_accuracy_62,1%.json', 'r')
json = f.read()
f.close()
model = tf.keras.models.model_from_json(json)

#Load model weights
model.load_weights(path + 'model_accuracy_62,1%.h5')