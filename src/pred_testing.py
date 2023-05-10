# pred_testing.py 
# 
# Script for prediction testing.
# 
# To run call:
# $ python pred_testing.py path/to/data/folder/ path/to/model/folder/
# 
# Author: Arnold Teo
# Version: 2023-04-21
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import os
import sys

model_type = 'vgg16'
model_name  = ''

if len(sys.argv) < 2:
    print('_______________________________________________________')
    print('Please type model name along with call.')
    print('Example call:    python pred_testing.py path/to/data/folder/ path/to/model/folder/')
    print('_______________________________________________________')
    sys.exit()
else:
    model_name  = sys.argv[2]

file_path = sys.argv[1] + 'testing/'
directory_files = os.listdir(file_path)
multiple_images = [file for file in directory_files if file.endswith(('.jpg', '.png'))]

#print(directory_files)
#print(multiple_images)

pred_list = []

for filename in multiple_images:
    img = cv2.imread(file_path+filename)
    img = preprocess_input_vgg16(img)
    pred_list.append(img)

model = load_model(model_name)

preds = model.predict(np.asarray(pred_list))

def pred_class(preds):
    classif = ''
    pred_val = np.round(preds)
    if len(pred_val) == 4:
        if pred_val[0] == 1:
            classif = 'corr'
        elif pred_val[1] == 1:
            classif = 'empty'
        elif pred_val[2] == 1:
            classif = 'up'
        elif pred_val[3] == 1:
            classif = 'down'
    elif len(pred_val) == 3:
        if pred_val[0] == 1:
            classif = 'plate'
        elif pred_val[1] == 1:
            classif = 'empty'
        elif pred_val[2] == 1:
            classif = 'closed'
    else:
        if pred_val[0] == 0:
            classif = 'corr'
        else:
            classif = 'incorr'
    return classif

print('_______________________________________________________')
for i in range(len(multiple_images)):
    print('File: {:^35}  Classification: {:^5}  One-hot: {}  Prediction: {}'.format(multiple_images[i], pred_class(preds[i]), np.round(preds[i]), preds[i]))
print('_______________________________________________________')
