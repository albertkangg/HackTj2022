import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model

temp = ['Blazer', 'Blouse', 'Cardigan', 'Jacket', 'Sweater', 'Tank', 'Tee', 'Top','Dress', 'Jumpsuit', 'Romper']
classes_1 = dict()
for i in range(len(temp)):
    classes_1[i] = temp[i]
temp = ['Jeans', 'Shorts', 'Skirt', 'Sweatpants','Dress', 'Jumpsuit', 'Romper']
classes_2 = dict()
for i in range(len(temp)):
    classes_2[i] = temp[i]

model_1 = keras.models.load_model(r'C:\Users\Kevin\HackTJ2022\model_1')
model_2 = keras.models.load_model(r'C:\Users\Kevin\HackTJ2022\model_2')

img_path = r'C:\Users\Kevin\HackTJ2022\DeepFashion\Test\Jeans\img_00000052.jpg'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

predict_1 = model_1.predict(x)
predict_1 = np.argmax(predict_1[0])
top_label = classes_1[predict_1]
print(top_label)

predict = model_2.predict(x)
predict = np.argmax(predict[0])
bottom_label = classes_2[predict]
print(bottom_label)