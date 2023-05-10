# ateo_mex
Code for training DNNs models to classify microwell plate classifications.

Done as part of KTH Master thesis work VT2023 by Arnold Teo (arnold_teo@hotmail.com) at Sandberg Lab KI.

Master thesis: *coming soon*

Sandberg GitHub: https://github.com/sandberg-lab/

Special packages required for running are:
```
pip install opencv-python-headless==4.7.0.68
pip install tensorflow==2.11.0
pip install matplotlib==3.6.3
pip install numpy==1.24.1
pip install scipy==1.10.0
pip install pydot==1.4.2
pip install winrt==1.0.21033.1 
```
Install graphviz according to: https://graphviz.gitlab.io/download/


## model_trainer_cycler.py
Creates, trains and saves a multi-class transfer learning model with retrained top layers. 
The data path and model parameters must be specified in the code by changing the `train_path` and `valid_path` variables.

Figures of the trained model architecture, the model loss curve, accuracy curve precision curve and recall curve will be saved in the `figs/train_plots/` folder. The trained model will be saved in the `src/models/` folder. 

**Before training**, make sure data is cropped and in the proper file structure:
```
                                data/
                                 |
                     multiclass_cycler_data_folder/
                    /                             \
                 train/                          valid/
        /      /       \       \       /       /       \      \
     corr/   down/   empty/   up/    corr/   down/   empty/   up/
```
The name of the `multiclass_cycler_data_folder/` folder does not matter, but subfolders **must** follow this name and structure.

Running model_trainer_cycler.py
```
python model_trainer_cycler.py
```


## model_trainer_dispenser.py
Creates, trains and saves a mutli-class transfer learning model with retrained top layers. 
The data path and model parameters must be specified in the code by changing the `train_path` and `valid_path` variables.

Figures of the trained model architecture, the model loss curve, accuracy curve precision curve and recall curve will be saved in the `figs/train_plots/` folder. The trained model will be saved in the `src/models/` folder. 

**Before training**, make sure data is cropped and in the proper file structure:
```
                                 data/
                                  |
                     multiclass_dispenser_data_folder/
                    /                             \
                 train/                          valid/
        /      /       \       \       /       /       \      \
     corr/   down/   empty/   up/    corr/   down/   empty/   up/
```
The name of the `multiclass_dispenser_data_folder/` folder does not matter, but subfolders **must** follow this name and structure.

Running model_trainer_dispenser.py
```
python model_trainer_dispenser.py
```


## model_trainer_centrifuge.py
Creates, trains and saves a multi-class transfer learning model with retrained top layers. 
The data path and model parameters must be specified in the code by changing the `train_path` and `valid_path` variables.

Figures of the trained model architecture, the model loss curve, accuracy curve precision curve and recall curve will be saved in the `figs/train_plots/` folder. The trained model will be saved in the `src/models/` folder. 

**Before training**, make sure data is cropped and in the proper file structure:
```
                           data/
                             |
                multiclass_centrifuge_data_folder/
                /                          \
             train/                       valid/
        /      |       \             /      |       \
     plate/  empty/   closed/     plate/  empty/   closed/
```
The name of the `multiclass_centrifuge_data_folder/` folder does not matter, but subfolders **must** follow this name and structure.

Running model_trainer_centrifuge.py
```
python model_trainer_centrifuge.py
```

## heatmap.py
Predicts with given model on input and creates a figure with heatmap overlay. The figure will be saved under `figs/train_plots/`.

Running heatmap.py example
```
python heatmap.py path/to/image path/to/model/folder
```

## pred_testing.py
Will predict on given testing data with given DNN model and print output in the terminal. The data folder structure must be as follows:
```
      data_folder/
          |
       testing/
```

Running pred_testing.py example
```
python pred_testing.py path/to/data_folder/ path/to/model/folder/
```


# Help scripts
## croppper.py
Crops all images in a training folder. Folder needs to be specified in code and raw images must be moved (and afterwards removed) manually.

Running croppper.py example
```
python croppper.py
```

## reposition_dispenser_cam.py, reposition_cycler_cam1.py and reposition_cycler_cam2.py
Script to help with repositioning of cameras. Will stream a video with overlay of a reference image. Press `ESC` key to exit. To change the camera IDs update the `instrument_id` variable.

Running reposition_dispenser_cam.py
```
python reposition_dispenser_cam.py
```

Running reposition_cycler_cam1.py
```
python reposition_cycler_cam1.py
```

Running reposition_cycler_cam2.py
```
python reposition_cycler_cam2.py
```

# src/implementation
## RobotVision.py
Class which captures and crops images. On start the class will capture an image each with each connected camera; dispenser, cyclers and centrifuge.

If cameras are changed new IDs will need to be set, these are found by printing the `cam_names` and `cam_ids` variables and updating the corresponding ID variables, for example `cycler_camera1_id`.  If camera postions are shifted, the crop coordinates might need to be updated. To do this change the `self.cen_x_instrument` and `self.cen_y_instrument` variables for the relevant instument(s). To find the new coordinates look at a `instrument_init.png` image from starting the automated lab with for example MATPLOTLIB.  

More detailed descriptions of class functions are in the thesis report.

## RobotBrain.py
Class which reads and predict on images. On start the class will read in the relevant DNN models. To update these change the path given in the relevant `load_model('path/to/model/folder/')` call. 

More detailed descriptions of class functions are in the thesis report.
