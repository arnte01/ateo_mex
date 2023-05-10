# model_trainer_centrifuge.py
#
# creates a transfer model with shallow NN on top for image classification of
# centrifuge plate placement
#
# To run call:
# $ python model_trainer_centrifuge.py path/to/data/folder/
#
# Author: Arnold Teo
# Version: 2023-04-21

# GPU setup
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

fix_gpu()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import sys
import matplotlib.pyplot as plt

# model details
img_width = 224
img_height = 224
img_ch = 3
base_dense = 4096
batch_sz = 30       # change the batch size depending on how much data there is
lr = 1e-5
momen=0.9
epoch_num = 50

retrain = False
shuffl = True

img_path = '../figs/train_plots/'

# model application vgg16
keras_app = tf.keras.applications.vgg16
imported_model = keras_app.VGG16
model_name = 'centrifuge'

# set paths for training and validation data
data_path = sys.argv[1]
train_path = data_path+'train'
valid_path = data_path+'valid'

# create data generator for image preprocessing
image_datagen = ImageDataGenerator(
    preprocessing_function=keras_app.preprocess_input,
        rotation_range = 10  
    )
# create data generator for training
train_batches = image_datagen.flow_from_directory(
        directory=train_path, 
        target_size=(img_width,img_height), 
        classes=['plate', 'empty', 'closed'], 
        class_mode = 'categorical',
        batch_size=batch_sz,
        shuffle = shuffl
    )
# create data generator for validation
valid_batches = image_datagen.flow_from_directory(
        directory=valid_path, 
        target_size=(img_width,img_height), 
        classes=['plate', 'empty', 'closed'], 
        class_mode = 'categorical',
        batch_size=batch_sz,
        shuffle = shuffl
    )

# import pretrained neural network without top dense layers for transfer learning
base_model = imported_model(
    include_top=False, 
    input_shape=(img_width, img_height, img_ch),
    weights='imagenet'
)
# set if imported model layers should be retrained or not
base_model.trainable = retrain

#base_model.summary()

# set input sizes
inputs = Input(shape=(img_width, img_height, img_ch))
base_temp = base_model(inputs, training=retrain)
# create NN model connected to imported model
in_layer = Flatten()(base_temp)
hidden_layer = Dense(base_dense//2, activation = 'relu')(in_layer)
dropout_layer = Dropout(0.5)(hidden_layer)
out_layer = Dense(3, activation = 'softmax')(dropout_layer)

# create transfer learning model
model = Model(inputs=inputs, outputs=out_layer) 
model.summary()

# train new model 
model.compile(
    optimizer=SGD(learning_rate=lr, momentum=momen), 
    loss=keras.losses.CategoricalCrossentropy(), 
    metrics=[keras.metrics.CategoricalAccuracy(), 
             keras.metrics.Precision(), 
             keras.metrics.Recall()
            ]
)
model_hist = model.fit(
    x=train_batches,
    steps_per_epoch=len(train_batches),
    validation_data=valid_batches,
    validation_steps=len(valid_batches),
    epochs=epoch_num,
    verbose=2
)
# save the model to disk
model.save('models/transfer_model_'+model_name)


# save plot of model layout
temp_name_str = img_path+'transfer_model_'+model_name+'.png'
keras.utils.plot_model(model, to_file=temp_name_str, show_shapes=True, show_dtype=True)

# plot loss curve
plt.figure(1)
plt.title("(a) Loss curve, categorical crossentropy")
plt.plot(model_hist.history['loss'], label = 'loss')
plt.plot(model_hist.history['val_loss'], label = 'val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.legend()
plt.savefig(img_path+'loss_'+model_name+'.jpg')

# plot accuracy curve
plt.figure(2)
plt.title("(b) Accuracy curve")
plt.plot(model_hist.history['categorical_accuracy'], label = 'categorical_accuracy')
plt.plot(model_hist.history['val_categorical_accuracy'], label = 'val_categorical_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(img_path+'acc_'+model_name+'.jpg')

# plot precision curve
plt.figure(3)
plt.title("(c) Precision curve")
plt.plot(model_hist.history['precision'], label = 'precision')
plt.plot(model_hist.history['val_precision'], label = 'val_precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.savefig(img_path+'precision_'+model_name+'.jpg')

# plot recall curve
plt.figure(4)
plt.title("(d) Recall curve")
plt.plot(model_hist.history['recall'], label = 'recall')
plt.plot(model_hist.history['val_recall'], label = 'val_recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.savefig(img_path+'recall_'+model_name+'.jpg')

# plot all curves in one
loss = plt.imread(img_path+'loss_'+model_name+'.jpg')
acc = plt.imread(img_path+'acc_'+model_name+'.jpg')
prec = plt.imread(img_path+'precision_'+model_name+'.jpg')
recall = plt.imread(img_path+'recall_'+model_name+'.jpg')
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 9))
ax = axes.ravel()
ax[0].imshow(loss)
ax[1].imshow(acc)
ax[2].imshow(prec)
ax[3].imshow(recall)
for a in ax:
    a.axis('off')
plt.suptitle('VGG16 transfer model {}\nOptimizer = SGD, Momentum = {}, Epochs = {}, LR = {}, Batch sz = {}'.format(model_name, momen, epoch_num, lr, batch_sz))
plt.savefig(img_path+'all_training_plots_'+model_name+'.jpg')
plt.show()
