# RobotBrain.py
#
# For dispenser, Cyclers and centrifuge 
#
# For multiclass Dispenser and Cyclers
# Predicts on images and classifies whether plates are placed:
# correctly     [1 0 0 0]
# empty         [0 1 0 0]
# up one well   [0 0 1 0]
# down one well [0 0 0 1]
#
# For multiclass Centrifuge
# Predicts on images and classifies whether plates are placed:
# plate is in centrifuge [1 0 0]
# Centrifuge is empty    [0 1 0]
# Centrifuge is closed   [0 0 1]
#
# Author: Arnold Teo
# Version 2023-05-02
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from datetime import datetime
import cv2, os, glob
import numpy as np

class RobotBrain:

    def __init__(self, protocol_path, slackbot):
        # connect slackbot
        self.slackbot = slackbot

        # initialise RobotArm as not connected
        self.connected_RobotArm = False

        # find latest image folder
        latest_folders = glob.glob('pictures/*/') 
        self.latest_folder = sorted(latest_folders, key=os.path.getctime, reverse=True)
        #print(self.latest_folder[0])

        # create an output .txt file for predictions
        self.outputfile = self.latest_folder[0]+'predictions.txt'
        f = open(self.outputfile, 'a')
        f.write('Predictions for protocol: '+protocol_path[12:]+'\nStarted on '+self.latest_folder[0][9:19]+' at '+self.latest_folder[0][20:22]+':'+self.latest_folder[0][23:25])
        f.write('\nFor dispenser and Cyclers\nCorrect multiclass placement predictions         = [1 0 0 0]')
        f.write('\nIncorrect multiclass placement predictions empty = [0 1 0 0]\nIncorrect multiclass placement predictions up    = [0 0 1 0]\nIncorrect multiclass placement predictions down  = [0 0 0 1]')
        f.write('\n\nFor centrifuge\nPlate is in centrifuge = [1 0 0]\ncentrifuge is empty    = [0 1 0]\ncentrifuge is closed   = [0 0 1]')
        f.write('\n_______________________________________________________')
        f.close()

        # import DNN image-classification models
        # binary models (removed)
        #self.binary_dispenser_model = load_model()
        #self.binary_cycler_model = load_model()
        #self.binary_centrifuge_model = load_model()
        # multiclass model
        self.multiclass_dispenser_model = load_model('devices/models/transfer_model_multiclass_dispenser-v5')
        self.multiclass_cycler_model = load_model('devices/models/transfer_model_multiclass_cycler-v2')
        self.multiclass_centrifuge_model = load_model('devices/models/transfer_model_multiclass_centrifuge-v2')


    def connect_RobotArm(self, RobotArm):
        self.RobotArm = RobotArm
        self.connected_RobotArm = True
        #print('RobotArm Connected')

    def prediction(self, instrument_type, camera_on=True):

        # if the camera is off assume placement is correct and return
        if not camera_on:
            # print predictions in the output .txt file
            f = open(self.outputfile, 'a')
            f.write('\nCamera off for {}, assuming correct placement.'.format(instrument_type))
            f.write('\n_______________________________________________________')
            f.close()
            return [0, 0, 0, 0]

        # find the latest 10 images in the folder
        images = glob.glob(self.latest_folder[0]+'*.png')
        latest_images = sorted(images, key=os.path.getctime, reverse=True)
        
        pred_list = []
        #preds = []
        preds_multi = []
        
        # read latest ten images
        for i in range(10):
            img = cv2.imread(latest_images[i])
            # preprocess input according to model
            img = preprocess_input(img)
            # model expects data to be list of images
            pred_list.append(img)

        # model predicts on images 
        if instrument_type.lower() == 'dispenser':      
            #preds = self.binary_dispenser_model.predict(np.asarray(pred_list))
            preds_multi = self.multiclass_dispenser_model.predict(np.asarray(pred_list))
        elif instrument_type.lower() == 'centrifuge':
            preds_multi = self.multiclass_centrifuge_model.predict(np.asarray(pred_list))
        elif instrument_type[0:6].lower() == 'cycler':
            #preds = self.binary_cycler_model.predict(np.asarray(pred_list))
            preds_multi = self.multiclass_cycler_model.predict(np.asarray(pred_list))
        
        # round prediction value and cast to Integer
        #ret_val = int(np.round(np.sum(preds)/10))
        ret_val_mult = np.round(np.sum(preds_multi, axis = 0)/10).astype(int)
        # if result is rounded to more than one class, assume incorrect
        if np.any(preds_multi) and np.sum(ret_val_mult) != 1:
            ret_val_mult = [0, 1, 0, 0]

        # print predictions in the output .txt file
        f = open(self.outputfile, 'a')
        #f.write('\nBinary predictions for '+instrument_type+':\n{}\nMajority voting value: {}'.format(preds, ret_val))
        f.write('\nMulticlass predictions for {}:\n{}\nMajority voting value: {}'.format(instrument_type, preds_multi, ret_val_mult))
        f.write('\n_______________________________________________________')
        f.close()
        
        return ret_val_mult

    def send_message(self, message):
        print(message)
        self.slackbot.post_msg(message)

    def stop_robot(self, message):
        # send Slack message and halt pipeline while waiting for user input
        self.slackbot.alert_msg(message)
        print(message)
        input('Press ENTER to continue protocol\n')

    def arm_reposition_cycler(self, mm, instrument_type):
        # movelist in the form
        # [ [fetch_movelists] , [return_movelists] ]
        json_cmds = self.RobotArm.get_movelists(instrument_type)
        z_disp = 4

        # ensure robot arm is in neutral position
        self.RobotArm.move_neutral()

        # make sure gripper is open
        self.RobotArm._open_gripper()
        # move into fetch position
        for cmds in json_cmds[0]:
            self.RobotArm.move_json(cmds)
        # relative to fetch position, move mm displacement and in z-axis
        corr_cmd = {"cmd":"lmove","rel":1,"x":mm,"y":0,"z":z_disp,"a":0,"b":0,"c":0}
        self.RobotArm.move_json(corr_cmd)
        # grab plate
        self.RobotArm._close_gripper()
        # move back to what should be proper placement
        corr_cmd = [
            {"cmd":"lmove","rel":1,"x":0,"y":0,"z":z_disp,"a":0,"b":0,"c":0},
            {"cmd":"lmove","rel":1,"x":-mm,"y":0,"z":0,"a":0,"b":0,"c":0},
            {"cmd":"lmove","rel":1,"x":0,"y":0,"z":-z_disp,"a":0,"b":0,"c":0}
            ]
        for cmd in corr_cmd:
            self.RobotArm.move_json(cmd)
        # release plate
        self.RobotArm._open_gripper()
        # return from cycler
        for cmds in json_cmds[1]:
            self.RobotArm.move_json(cmds)
        # move arm out of the way for image classification
        if instrument_type[6:] in ['10', '11']:
            self.RobotArm.move_home()
        else: 
            self.RobotArm.move_neutral()


    def arm_reposition_dispenser(self, mm, instrument_type):  
        # movelist in the form
        # [ [fetch_movelists] , [return_safe_movelists] ]
        json_cmds = self.RobotArm.get_movelists(instrument_type)  

        # ensure robot arm is in neutral position
        self.RobotArm.move_neutral()

        # read in coordinates to find angle of gripper placment
        x1 = json_cmds[0][-3]['x']
        x2 = json_cmds[0][-2]['x']
        y1 = json_cmds[0][-3]['y']
        y2 = json_cmds[0][-2]['y']
        # calculate distance in movement into dispenser plate position
        delta_x = x1-x2
        delta_y = y1-y2
        # calculate the angle
        alpha = np.arctan(delta_x / delta_y)
        # calculate correction in x-axis and y-axis
        x_disp = mm * np.sin(alpha)
        y_disp = mm * np.cos(alpha)
        z_disp = 4

        # make sure gripper is open
        self.RobotArm._open_gripper()
        # move into fetch position
        for cmds in json_cmds[0]:
            self.RobotArm.move_json(cmds)
        # relative to fetch position, move one well displacement and in z-axis
        corr_cmd = {"cmd":"lmove","rel":1,"x":x_disp,"y":y_disp,"z":z_disp,"a":0,"b":0,"c":0}
        self.RobotArm.move_json(corr_cmd)
        # grab plate
        self.RobotArm._close_gripper()
        # move back to what should be proper placement
        corr_cmd = [
            {"cmd":"lmove","rel":1,"x":0,"y":0,"z":2*z_disp,"a":0,"b":0,"c":0},
            {"cmd":"lmove","rel":1,"x":-x_disp,"y":-y_disp,"z":0,"a":0,"b":0,"c":0},
            {"cmd":"lmove","rel":1,"x":0,"y":0,"z":-2*z_disp,"a":0,"b":0,"c":0}
            ]
        for cmd in corr_cmd:
            self.RobotArm.move_json(cmd)
        # release plate
        self.RobotArm._open_gripper()
        # return from dispenser
        for cmds in json_cmds[1]:
            self.RobotArm.move_json(cmds)
        # move back to neutral position
        self.RobotArm.move_neutral()

    def verify_placement(self, mult_pred_class, instrument_type, reposition=True, camera_on=True):

        # if the camera is off assume proper placement and return
        print('_______________________________________________________') 
        if not camera_on:
            print('Camera off, assuming correct placement on {}.'.format(instrument_type))
            correct_placement = True
            err_message = 'Camera off, assuming correct placement.'
            repositioning = False
            return correct_placement, err_message, repositioning    

        correct_placement = False
        repositioning = False
        well_dist = 4.5

        classif = ''
        err_message = 'Error: Incorrect placement on '+instrument_type

        if np.any(mult_pred_class) and instrument_type.lower() == 'centrifuge':
            classif = pred_class_centrifuge(mult_pred_class)
            if classif != 'plate':
                err_message = 'Error: centrifuge is '+classif
                return correct_placement, err_message, repositioning 
            else:
                print('Plate placed in  '+instrument_type)
                correct_placement = True
                return correct_placement, err_message, repositioning 

               
        #print('Predicted binary class: '+str(bin_pred_class))
        if np.any(mult_pred_class):
            classif = pred_class(mult_pred_class)
            print('Predicted multip class: {}  {}'.format(classif, str(mult_pred_class)))
            
        if classif != 'corr':
            if self.connected_RobotArm and reposition and classif == 'up':
                repositioning = True
                # try to reposition plate if misplaced one well 'up' if viewed from the front of the instrument
                err_message = 'Error: RobotArm tried to reposition plate on {} but failed.'.format(instrument_type)
                if instrument_type.lower() == 'dispenser':
                    self.arm_reposition_dispenser(-well_dist, instrument_type)
                elif instrument_type[0:6].lower() == 'cycler':
                    self.arm_reposition_cycler(well_dist, instrument_type)
                else:
                    err_message = 'Error: Incorrect placement on '+instrument_type      
            elif self.connected_RobotArm and reposition and classif == 'down':
                repositioning = True
                # try to reposition plate if misplaced one well 'down' if viewed from the front of the instrument
                err_message = 'Error: RobotArm tried to reposition plate on {} but failed.'.format(instrument_type)
                if instrument_type.lower() == 'dispenser':
                    self.arm_reposition_dispenser(2*well_dist, instrument_type)
                elif instrument_type[0:6].lower() == 'cycler':
                    self.arm_reposition_cycler(-well_dist, instrument_type)
                else:
                    err_message = 'Error: Incorrect placement on '+instrument_type
            else:
                pass        

        elif classif == 'corr':
            print('Correct placement on  '+instrument_type)
            correct_placement = True

        return correct_placement, err_message, repositioning

def pred_class_centrifuge(preds):
    # function to decode the multiclass prediction output for centrifuge
    classif = ''
    pred_val = np.round(preds)
    if pred_val[0] == 1:
        classif = 'plate'
    elif pred_val[1] == 1:
        classif = 'empty'
    elif pred_val[2] == 1:
        classif = 'closed'
    return classif

def pred_class(preds):
    # function to decode the multiclass prediction output for dispenser and Cycler
    classif = ''
    pred_val = np.round(preds)
    if pred_val[0] == 1:
        classif = 'corr'
    elif pred_val[1] == 1:
        classif = 'empty'
    elif pred_val[2] == 1:
        classif = 'up'
    elif pred_val[3] == 1:
        classif = 'down'
    return classif
