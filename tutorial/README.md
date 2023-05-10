# Tutorial
Tutorial for training one of the DNN models, in this case the multiclass nano-dispenser model. For training of other models the overall process is the same.

Make sure you have installed all the required software stated in the main README file.

## Data collection
For the purposes of the tutorial the data is already present under the `dispenser_example` folder. If you are training a completely new model look below.

Collect data by running `takepic.py` or the automated lab. To run the `takepic.py` script do it as:
```
$ python takepic.py folder/to/save/images/in/ Dispenser 
```
Move the training data to the `data/` directory and place it in a new folder. In this new folder, create three new directories; `train/`,  `valid` and `testing/`. In each of the `train/` and `valid/` folders, create the folders; `corr`, `empty`, `up`, `down`. Place collected data (around 100-150 images per class) in above folders, 70-80% of images in each training folder, and around 20-30% of images in each validation folder. In the `testing/` folder place 10-20 images for prediction testing. Note that there should **not** be any duplicated images between the `train/`,  `valid/` and `testing/` directories.

If the training data is not cropped to a size of 224x224 pixels, change the `file_path` variable in the `cropper.py` script to the directory with the training or validation data. Run `cropper.py` once for the `train/` directory, and once for the `valid/` directory. Have a look and see if the images look correct. From the folders, remove the larger, raw images (the cropped files will have a prefix which says `crp_`). The raw data can be deleted or moved to a different folder, as long as it is not in the `train/` or `valid/` folders. 
  
## Training
Open the terminal in Python virtual environment or Conda. Move to the `src/` folder. Run the dispenser trainer with:
```
$ python model_trainer_dispenser.py ../data/dispenser_tutorial/
```
Once the training is completed there will be six figures saved under the `figs/train_plots/` directory; a figure of the model architecture, loss plot, accuracy plot, precision plot, recall plot and a figure with the previous four plots combined. 

The model is saved in a folder under the `src/models/` directory under the name `transfer_model_dispenser`, if you wish to you can change the folder name.

## Validation
Once the model is trained, the model heatmap can be created using the following command (if you haven't changed the model name the path from the `src/dispenser/` directory should be `models/transfer_model_dispenser`):
```
$ python heatmap.py ../data/dispenser_tutorial/testing/testing_corr1.png models/transfer_model_multiclass_dispenser/
```
The heatmap figure will be stored under `figs/train_plots/`. All the figures here can then be moved to a different folder of choice, if not, they will be overwritten when a new model is trained. 

Prediction testing can be done by running:
```
$ python pred_testing.py ../data/dispenser_tutorial/ models/transfer_model_multiclass_dispenser/
```
This will print result in the terminal which will need to be verified manually. A tip is to name your testing data according to what the correct prediction should be!

Use the heatmap, test prediction and training plots to validate if the model perfomance is good. 

## Implementation
Add the new model to the `RobotBrain.py` script by changing the `self.multiclass_dispenser_model()` variable. To do this change the path to the new model declared in `load_model('path/to/model/transfer_model_multiclass_dispenser')`.

Congratulations! You have now implemented a new nano-dispenser plate placement classification model into the automated lab.
