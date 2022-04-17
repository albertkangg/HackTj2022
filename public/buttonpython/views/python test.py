import sys
import base64
from PIL import Image
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

image_path=sys.argv[1]
image_name=sys.argv[2]
image_save_path=image_path.replace(image_name,"temp.png")
img = Image.open(str(image_path))
img.save(image_save_path)
# print("/media/temp.png")


temp = ['Blazer', 'Blouse', 'Cardigan', 'Jacket', 'Sweater', 'Tank', 'Tee', 'Top','Dress', 'Jumpsuit', 'Romper']
classes_1 = dict()
for i in range(len(temp)):
    classes_1[i] = temp[i]
temp = ['Jeans', 'Shorts', 'Skirt', 'Sweatpants','Dress', 'Jumpsuit', 'Romper']
classes_2 = dict()
for i in range(len(temp)):
    classes_2[i] = temp[i]

model_1 = keras.models.load_model(r'C:\Users\inyuikang\HackTJ9\public\buttonpython\views\model_1')
model_2 = keras.models.load_model(r'C:\Users\inyuikang\HackTJ9\public\buttonpython\views\model_2')

img = image.load_img(image_save_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

predict_1 = model_1.predict(x)
predict_1 = np.argmax(predict_1[0])
top_label = classes_1[predict_1]

predict = model_2.predict(x)
predict = np.argmax(predict[0])
bottom_label = classes_2[predict]
print(image_save_path,top_label,bottom_label)
