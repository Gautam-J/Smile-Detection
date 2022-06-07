import numpy as np
import coremltools
from tensorflow.keras.models import load_model

model = load_model('./models/baseline.hdf5')
coreml_model = coremltools.converters.convert(
    model=model,
    source='tensorflow',
)

coreml_model.save('./models/smile_detection.mlmodel')

test_input = np.random.rand(1, 64, 64, 3)

orig_pred = model.predict(test_input)[0][0]
conv_pred = coreml_model.predict({'rescaling_input': test_input})
conv_pred = conv_pred['Identity'][0][0]

print("Original model output:", orig_pred)
print("Converted model output:", conv_pred)
