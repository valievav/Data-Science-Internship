import os
import pickle

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from parameters import img_model_name, class_mapping_filename, IMG_SIZE, BATCH_SIZE, SEED

# get data from kaggle
# https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals/code?datasetId=1554380&sortBy=voteCount
root_path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")

# path to the animals images folder
path = os.path.join(root_path, "animals", "animals")

# prepare general df with all the data for further split
category = os.listdir(path)

data = {"imgpath": [] , "labels": [] }
for folder in category:
    folderpath = os.path.join(path , folder)
    filelist = os.listdir(folderpath)
    for file in filelist:
        fpath = os.path.join(folderpath, file)
        data["imgpath"].append(fpath)
        data["labels"].append(folder)

df = pd.DataFrame(data)

# add column with encoded labels
lb = LabelEncoder()
df['encoded_labels'] = lb.fit_transform(df['labels'])

# prepare label/class lookup
class_mapping = dict(zip(map(int, lb.transform(lb.classes_)), lb.classes_))

# save results into pickle file for reuse
with open(class_mapping_filename, "wb") as f:
    pickle.dump(class_mapping, f)

# visualize 1st image
index = 0
img = Image.open(df['imgpath'][index])
plt.imshow(img)
plt.title(df["labels"][index])
plt.axis('off')
plt.show()

# visualize 8 random images
n = 8
plt.figure(figsize=(15,12))
for i, row in df.sample(n=8).reset_index().iterrows():
    plt.subplot(4,4,i+1)
    image_path = row['imgpath']
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(row["labels"])
    plt.axis('off')
plt.show()

# split into 3 df
train_df, Temp_df = train_test_split(df,  train_size= 0.70 , shuffle=True, random_state=124)
valid_df , test_df = train_test_split(Temp_df ,  train_size= 0.70 , shuffle=True, random_state=124)

# remove old index columns
train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

generator = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.efficientnet.preprocess_input,
    rescale=1./255,  # normalize pixel values to [0, 1]
)

# split the data into 3 categories
train_ds = generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='imgpath',
    y_col='labels',
    target_size=IMG_SIZE,
    color_mode='rgb',
    class_mode='sparse',  # since labels are not one-hot encoded, else categorical
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = generator.flow_from_dataframe(
    dataframe=valid_df,
    x_col='imgpath',
    y_col='labels',
    target_size=IMG_SIZE,
    color_mode='rgb',
    class_mode='sparse',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

test_ds = generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='imgpath',
    y_col='labels',
    target_size=IMG_SIZE,
    color_mode='rgb',
    class_mode='sparse',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

# use MobileNetV2 (light-wight model, pre-trained on ImageNet)
base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')

# freeze base model
base_model.trainable = False

# build model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(len(train_ds.class_indices), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # since labels are not one-hot encoded, else categorical_crossentropy
              metrics=['accuracy'])

model.summary()

# train model
history = model.fit(train_ds, validation_data=val_ds, epochs=5)

# evaluate model on test data
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

# save model for future use
model.save(img_model_name)
