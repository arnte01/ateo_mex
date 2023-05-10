# heatmap.py
# 
# Script to generate a heatmap with CNN model
#
# To run call:
# $ python heatmap.py path/to/image path/to/model/folder/
#
# Author: Arnold Teo
# Version: 2023-04-28
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import os
import sys

from tensorflow.keras import backend as K
import tensorflow as tf

model_type = 'vgg16'
model_name = ''
if len(sys.argv) < 2:
    print('_______________________________________________________')
    print('Please type model name along with call.')
    print('Example call:    python heatmap.py path/to/image path/to/model/folder/')
    print('_______________________________________________________')
    sys.exit()
else:
    model_name = sys.argv[2]

file = sys.argv[1]

img = cv2.imread(file)
img_raw = img

pred_list = []
img = preprocess_input_vgg16(img_raw)

pred_list.append(img)

model = load_model(model_name)

preds = model.predict(np.asarray(pred_list))
print('Prediction output: {}'.format(preds))

def heatmap_gen(pred_list, img_raw):
    # edited from 
    # https://medium.com/analytics-vidhya/visualizing-activation-heatmaps-using-tensorflow-5bdba018f759
    with tf.GradientTape() as tape:
        model_type = 'vgg16'
        layer_name = 'block5_conv3'
        heatmap_dim = 14

        last_conv_layer = model.get_layer(model_type).get_layer(layer_name)
        iterate = tf.keras.models.Model([model.get_layer(model_type).get_layer('input_1').input], [model.get_layer(model_type).output, last_conv_layer.output])
            
        model_out, last_conv_layer = iterate(np.asarray(pred_list))

        grads = tape.gradient(model_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((heatmap_dim, heatmap_dim))

    heatmap = cv2.resize(heatmap, (img_raw.shape[1], img_raw.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    #img = heatmap * intensity + img
    superimposed_img = cv2.addWeighted(img_raw, 0.6, heatmap, 0.4, 0)

    return superimposed_img

heatmap = heatmap_gen(pred_list, img_raw)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
ax = axes.ravel()
ax[0].imshow(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
ax[0].set_title('Input image')
ax_tmp = ax[1].imshow(heatmap)
plt.colorbar(ax_tmp)
ax[1].set_title('Heatmap')

for a in ax:
    a.axis('off')
plt.suptitle('Heatmap from final {} convolution layer.'.format(model_type))
plt.savefig('../figs/train_plots/heatmap_{}.png'.format(model_type))
plt.show()

