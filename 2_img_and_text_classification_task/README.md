**Task 2. Named entity recognition + image classification** 

In this task, you will work on building your ML pipeline that consists of 2 models responsible for
totally different tasks. The main goal is to understand what the user is asking (NLP) and check if
he is correct or not (Computer Vision).

You will need to:
* find or collect an animal classification/detection dataset that contains at least 10
classes of animals.
* train NER model for extracting animal titles from the text. Please use some
transformer-based model (not LLM).
* Train the animal classification model on your dataset.
* Build a pipeline that takes as inputs the text message and the image.

In general, the flow should be the following:
1. The user provides a text similar to “There is a cow in the picture.” and an image that
contains any animal.
2. Your pipeline should decide if it is true or not and provide a boolean value as the output.
You should take care that the text input will not be the same as in the example, and the
user can ask it in a different way.

The solution should contain:
* Jupyter notebook with exploratory data analysis of your dataset;
* Parametrized train and inference .py files for the NER model;
* Parametrized train and inference .py files for the Image Classification model;
* Python script for the entire pipeline that takes 2 inputs (text and image) and provides
1 boolean value as an output;

############################################

For **Image Classification** used existing **dataset from kaggle** that contains 90 different 
animal classes. For base model used **MobileNetV2** (light-weight model, pre-trained on ImageNet).

For **Text Classification** used custom dataset from **text_classification/prepare_dataset.py**, 
created for 10 animal classes. For base model used **distilbert-base-uncased** (light-weight model).

To rerun the whole process:
1. Run **image_classification/train.py**
2. Check results by running **image_classification/inference.py**
3. Run **text_classification/train.py**
4. Check results by running **text_classification/inference.py**

To run only user input prediction:
1. Download pretrained NER model using **[this link](https://drive.google.com/file/d/1piFIcSIhDCeo_byadtmTdi1NJvny7-vD/view?usp=drive_link)** and unzip it
(could not upload it to github because of the size - 3.9GB raw, 2.8GB zipped).
2. Run **main.py** to verify if animal from user text and image input is matching.

Important: 
1. Although image classification trained on 90 animal classes, 
text classification trained only on 10 (see *class_names* in *text_classification/parameters.py*), 
so if you'd like to change user input (text & image), please stick to animals from those 10 classes.
2. Text classification custom dataset is too small to have effective training results, but it's enough to prove the concept. 
If dataset source is switched to bigger one, model will perform much better. 
Currently model detects animal classes as expected, but there are instances, when it returns empty label prediction.
