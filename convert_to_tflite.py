import tensorflow as tf

model = tf.keras.models.load_model('models/baseline.hdf5')
tf.saved_model.save(model, 'saved_model')

converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
