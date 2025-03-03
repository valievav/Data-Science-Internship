import pickle
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from parameters import img_model_name, class_mapping_filename, IMG_SIZE

loaded_model = load_model(img_model_name)

url_mapping = {
    'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/800px-Cat_November_2010-1a.jpg': 'cat',
    'https://images.pexels.com/photos/96938/pexels-photo-96938.jpeg': 'cat',
    'https://as2.ftcdn.net/v2/jpg/06/65/77/15/1000_F_665771503_IugD47zc2ojqj7jrtVDNn5cusVEAr3LM.jpg': 'jellyfish',
    'https://pethelpful.com/.image/w_1080,q_auto:good,c_fill,ar_4:3/NDowMDAwMDAwMDAwMDYzOTIw/closeupofyoungseaotterenhydralutrisfloatinginocean.jpg': 'otter',
    'https://c7.alamy.com/comp/CW8CN8/scottish-highland-ox-CW8CN8.jpg': 'ox',
}

# get label mapping
with open(class_mapping_filename, "rb") as f:
    class_mapping = pickle.load(f)

# retrieve images from the web and access predictions
for url, actual_label in url_mapping.items():
  response = requests.get(url)
  img = Image.open(BytesIO(response.content))

  # show the image
  plt.imshow(img)
  plt.axis('off')  # hide axes
  plt.show()

  # preprocess the image for model
  img = img.resize(IMG_SIZE)  # resize to match model input size
  img_array = img_to_array(img)  # convert image to array (height, width, channels)
  img_array = np.expand_dims(img_array, axis=0)  # add batch dimension (1, height, width, channels)
  img_array = img_array / 255.0  # Normalize (0-1 range)

  # get predictions
  predictions = loaded_model.predict(img_array)
  predicted_class = np.argmax(predictions, axis=-1) # get the class with the highest probability
  print(f"Predicted class: {predicted_class}")

  predicted_label = class_mapping[predicted_class[0]]
  correct = predicted_label == actual_label
  print(f"Predicted label: {predicted_label}. CORRECT? {correct}")
