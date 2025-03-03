import os
import pickle
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from transformers import pipeline
from image_classification.parameters import class_mapping_filename, img_model_name, IMG_SIZE
from text_classification.parameters import ner_model_name

os.environ["WANDB_DISABLED"] = "true"

text_model_path = os.path.join('text_classification', ner_model_name)
img_model_path = os.path.join('image_classification', img_model_name)
class_mapping_path = os.path.join('image_classification', class_mapping_filename)
img_loaded_model = load_model(img_model_path)


def get_label_prediction_image(img_url: str) -> str:
    """
    Return label prediction for image
    """
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))

    # preprocess the image for model
    img = img.resize(IMG_SIZE)  # resize to match model input size
    img_array = image.img_to_array(img)  # convert image to array (height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension (1, height, width, channels)
    img_array = img_array / 255.0  # Normalize (0-1 range)
    print(img_array.shape)  # should be (1, 224, 224, 3)

    # get label mapping
    with open(class_mapping_path, "rb") as f:
        class_mapping = pickle.load(f)

    # get predictions
    predictions = img_loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)  # get the class with the highest probability
    predicted_label = class_mapping[predicted_class[0]]
    return predicted_label

def get_label_prediction_text(text: str) -> str:
    """
    Return label prediction for text
    """
    # Load the pipeline
    classifier = pipeline("ner", model=text_model_path, tokenizer=text_model_path)
    classifier.model.eval()

    # Inference
    results = classifier(text)
    predicted_label = results[0]['entity'].replace('B-', '').lower() if results else None # assume text can only have 1 animal
    return predicted_label


def get_user_input(sentence: str, image_url: str) -> bool:
    """
    Find label for user test & image and evaluate if they match.
    """
    text_predicted_label = get_label_prediction_text(sentence)
    img_predicted_label = get_label_prediction_image(image_url)

    match = img_predicted_label == text_predicted_label
    if match:
        print(f'CORRECT :) ! Image contains {text_predicted_label}')
    else:
        print(f'INCORRECT :( ! Image detected label {img_predicted_label}, text detected label {text_predicted_label}')

    return match

# evaluate user text and image input to detect if they are both about the same animal
# for this use trained models to detect animal class in a text and in a picture
url = 'https://cdn.mos.cms.futurecdn.net/jhVpZ596HMj8AXeny8ZrMA-1200-80.jpg'
text = "There's a caterpillar in this picture"
match = get_user_input(text, url)
print(match)
