import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

def DataLoad1(shape, preprocessing): 
    '''Create the training and validation datasets for 
    a given image shape.
    '''
    imgdatagen = ImageDataGenerator(
        preprocessing_function = preprocessing,
        horizontal_flip = True, 
        validation_split = 0.0,
    )

    height, width = shape

    train_dataset = imgdatagen.flow_from_directory(
        os.getcwd(),
        target_size = (height, width), 
        classes = ['Blazer', 'Blouse', 'Cardigan', 'Jacket', 'Sweater', 'Tank', 'Tee', 'Top','Dress', 'Jumpsuit', 'Romper'],
        batch_size = batch_size,
        subset = 'training', 
    )

    val_dataset = imgdatagen.flow_from_directory(
        os.getcwd(),
        target_size = (height, width), 
        classes = ['Blazer', 'Blouse', 'Cardigan', 'Jacket', 'Sweater', 'Tank', 'Tee', 'Top','Dress', 'Jumpsuit', 'Romper'],
        batch_size = batch_size,
        subset = 'validation'
    )
    return train_dataset, val_dataset

def DataLoad2(shape, preprocessing): 
    '''Create the training and validation datasets for 
    a given image shape.
    '''
    imgdatagen = ImageDataGenerator(
        preprocessing_function = preprocessing,
        horizontal_flip = True, 
        validation_split = 0.0,
    )

    height, width = shape

    train_dataset = imgdatagen.flow_from_directory(
        os.getcwd(),
        target_size = (height, width), 
        classes = ['Jeans', 'Shorts', 'Skirt', 'Sweatpants','Dress', 'Jumpsuit', 'Romper'],
        batch_size = batch_size,
        subset = 'training', 
    )

    val_dataset = imgdatagen.flow_from_directory(
        os.getcwd(),
        target_size = (height, width), 
        classes = ['Jeans', 'Shorts', 'Skirt', 'Sweatpants','Dress', 'Jumpsuit', 'Romper'],
        batch_size = batch_size,
        subset = 'validation'
    )
    return train_dataset, val_dataset

vgg16 = keras.applications.vgg16

datasetdir = r'C:\Users\Kevin\HackTJ2022\DeepFashion\Train/1'
os.chdir(datasetdir)
batch_size = 3
train_dataset_1, val_dataset_1 = DataLoad1((224,224), preprocessing=vgg16.preprocess_input)
x_train_1, y_train_1 = next(train_dataset_1)

datasetdir = r'C:\Users\Kevin\HackTJ2022\DeepFashion\Train/2'
os.chdir(datasetdir)
batch_size = 3
train_dataset_2, val_dataset_2 = DataLoad2((224,224), preprocessing=vgg16.preprocess_input)
x_train_2, y_train_2 = next(train_dataset_2)

conv_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
# flatten the output of the convolutional part: 
x = keras.layers.Flatten()(conv_model.output)
# three hidden layers
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
# final softmax layer with 15 categories
predictions = keras.layers.Dense(11, activation='softmax')(x)

# creating the full model:
full_model_1 = keras.models.Model(inputs=conv_model.input, outputs=predictions)
full_model_2 = keras.models.Model(inputs=conv_model.input, outputs=predictions)

for layer in conv_model.layers:
    layer.trainable = False

full_model_1.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adamax(learning_rate=0.001),
                  metrics=['acc'])

full_model_2.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adamax(learning_rate=0.001),
                  metrics=['acc'])

# train model_1
full_model_1.fit_generator(
    train_dataset_1, 
    validation_data = val_dataset_1,
    workers=0,
    epochs=5,
)

#train model_2
full_model_2.fit_generator(
    train_dataset_2, 
    validation_data = val_dataset_2,
    workers=0,
    epochs=5,
)

full_model_1.save(r'C:\Users\Kevin\HackTJ2022\model_1')
full_model_2.save(r'C:\Users\Kevin\HackTJ2022\model_2')
