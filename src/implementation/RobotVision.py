# RobotVision.py
#
# Captures image with webcamera 
# 
# Update coordinates if cameras have been moved by changing variables:
# @self.cen_x_dispenser
# @self.cen_y_dispenser
# @self.cen_x_cycler
# @self.cen_y_cycler
# @self.cen_x_centrifuge
# @self.cen_y_centrifuge
#
# If cameras are changed for the dispenser or centrifuge, update camera names in variables:
# @dispenser_camera_name
# @centrifuge_camera_name
#
# If the cycler cameras are changed, update the ID values in variables:
# @cycler_camera1_id
# @cycler_camera2_id
#
# Author: Arnold Teo
# Version: 2023-04-26
import cv2, time, os, sys
from datetime import datetime

import asyncio
import winrt.windows.devices.enumeration as windows_devices

async def get_camera_info():
    #from: https://stackoverflow.com/questions/73946689/get-usb-camera-id-with-open-cv2-on-windows-10
    return await windows_devices.DeviceInformation.find_all_async(4)

class RobotVision:

    def __init__(self, slackbot):
        # connect slackbot
        self.slackbot = slackbot

        # model output image size
        self.img_height = 224
        self.img_width = 224

        # coordinates for cropping
        self.cen_x_dispenser = 1045
        self.cen_y_dispenser = 604
        self.cen_x_cycler = [1545, 1220, 855, 1395]
        self.cen_y_cycler = [990,  980,  970, 1055]
        self.cen_x_centrifuge = 320
        self.cen_y_centrifuge = 240

        # create camera list
        self.cameras = []

        # create a folder for captured images with name yyyy-mm-dd_hh-mm
        folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.pic_folder = 'pictures/'+folder_name
        try:
            # if folder was already created this minute, delete it
            os.rmdir(self.pic_folder)
        except:
            pass
        os.mkdir(self.pic_folder)

        # find cameras
        # dispenser (Logitech) camera name:     'HD Pro Webcam C920'
        # Cycler    (Sandberg) camera name:     'Sandberg Pro Elite 4K UHD' 
        # centrifuge   (Intel) camera names:    'Intel(R) RealSense(TM) Depth Camera 435 with RGB Module RGB', 
        #                                       'Intel(R) RealSense(TM) Depth Camera 435 with RGB Module Depth' 
        dispenser_camera_name = 'HD Pro Webcam C920'
        centrifuge_camera_name = 'Intel(R) RealSense(TM) Depth Camera 435 with RGB Module RGB'

        # both cycler cameras are the same model and share the same names
        # therefore the IDs are required
        cycler_camera1_id = '\\\\?\\USB#VID_1BCF&PID_28C4&MI_00#7&17f36d47&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\\GLOBAL'
        cycler_camera2_id = '\\\\?\\USB#VID_1BCF&PID_28C4&MI_00#7&13d8f827&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\\GLOBAL'

        # list camera names and IDs
        connected_cameras = asyncio.run(get_camera_info())
        cam_ids = [camera.id for camera in connected_cameras]
        cam_names = [camera.name for camera in connected_cameras]
        
        # use these print functions to find camera names and IDs
        #print(cam_names)
        #print(cam_ids)
        
        # find camera device index for connection
        cycler_camera1_index = 99
        cycler_camera2_index = 99
        dispenser_camera_index = 99
        centrifuge_camera_index = 99
        if cycler_camera1_id in cam_ids:
            cycler_camera1_index = cam_ids.index(cycler_camera1_id)
        if cycler_camera2_id in cam_ids:
            cycler_camera2_index = cam_ids.index(cycler_camera2_id)
        if dispenser_camera_name in cam_names:
            dispenser_camera_index = cam_names.index(dispenser_camera_name)
        if centrifuge_camera_name in cam_names:
            centrifuge_camera_index = cam_names.index(centrifuge_camera_name)
        
        # connect camera stream
        camera_cycler1 = cv2.VideoCapture(cycler_camera1_index)
        camera_cycler2 = cv2.VideoCapture(cycler_camera2_index)
        camera_dispenser = cv2.VideoCapture(dispenser_camera_index)
        camera_centrifuge = cv2.VideoCapture(centrifuge_camera_index)

        # set resolution
        # Sandberg: 2048, 1536 (4K setting increases glare, do 1536p instead)
        # Logitech 1080p: 1920, 1080
        # Intel 480p: 640, 480
        camera_cycler1.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
        camera_cycler1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)

        camera_cycler2.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
        camera_cycler2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)

        camera_dispenser.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera_dispenser.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        camera_centrifuge.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera_centrifuge.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # set autofocus off
        camera_cycler1.set(cv2.CAP_PROP_AUTOFOCUS, 0)  
        camera_cycler2.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        camera_dispenser.set(cv2.CAP_PROP_AUTOFOCUS, 0)       
        camera_centrifuge.set(cv2.CAP_PROP_AUTOFOCUS, 0)  

        # set focus
        # between 0:255 in 5-step increments
        # 0 starts at the camera lens
        # Sandberg focus=10
        # Logitech focus=15
        # Intel focus=5
        camera_cycler1.set(cv2.CAP_PROP_FOCUS, 10)
        camera_cycler2.set(cv2.CAP_PROP_FOCUS, 10)
        camera_dispenser.set(cv2.CAP_PROP_FOCUS, 15)
        camera_centrifuge.set(cv2.CAP_PROP_FOCUS, 5)

        self.cameras.append(camera_cycler1)
        self.cameras.append(camera_cycler2)
        self.cameras.append(camera_dispenser)
        self.cameras.append(camera_centrifuge)

        # test that the cameras are working
        # if not, user will be prompted if they want to continue 
        # if not, the program exits, otherwise continues without camera validation
        camera_str = ['Cycler camera 1', 'Cycler camera 2', 'dispenser camera', 'centrifuge camera']
        for cam_idx in range(len(camera_str)):
            try:
                ret, frame = self.cameras[cam_idx].read()
                if ret:
                    print('{} initialised.'.format(camera_str[cam_idx]))
                    cv2.imwrite('{}/{}_init.png'.format(self.pic_folder, camera_str[cam_idx].replace(" ", "_")), frame)
                else:
                    raise Exception('{} not initialised properly.'.format(camera_str[cam_idx])) 
            except Exception as exception:
                print('_______________________________________________________\n\n!! WARNING: {} !!'.format(exception))
                usr_input = input('Continue anyway without {} validation?\n_______________________________________________________\n'.format(camera_str[cam_idx]))
                if usr_input[0].lower() == 'n':
                    sys.exit()
                else:
                    camera.release()


    def capture_image(self, instrument):
        plate_cen_y = 0
        plate_cen_x = 0

        kill_pipeline = False

        # set variables for frame capture name and path
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        #filename = 'pictures/'+dt_string+'.png'
        filename = self.pic_folder+'/'+dt_string+instrument

        # set camera index to capture with correct camera
        # set local variable for cropping images to region of interest
        cam_id = 99
        if instrument[0:6].lower() == 'cycler':            
            index = int(instrument[6:])-1
            plate_cen_y = self.cen_y_cycler[index]
            plate_cen_x = self.cen_x_cycler[index]

            # cycler camera 1 oversees cyclers 1, 2
            if (instrument[6:] in ['1', '2']):
                cam_id = 0
            else:
                cam_id = 1
        elif instrument.lower() == 'dispenser':
            plate_cen_y = self.cen_y_dispenser
            plate_cen_x = self.cen_x_dispenser
            cam_id = 2
        elif instrument.lower() == 'centrifuge':
            plate_cen_y = self.cen_y_centrifuge
            plate_cen_x = self.cen_x_centrifuge
            cam_id = 3

        time.sleep(1)

        # return if the camera is off
        if not self.cameras[cam_id].isOpened():
            return False, kill_pipeline

        # capture ten images
        for i in range(10):
            try:
                # capture frame
                ret, frame = self.cameras[cam_id].read()

                if ret:
                    # crop frame to output region of interest image
                    img = frame[(plate_cen_y-self.img_height//2):(plate_cen_y+self.img_height//2), (plate_cen_x-self.img_width//2):(plate_cen_x+self.img_width//2), :]

                    # save image
                    tmp = filename+'_'+str(i)+'.png'
                    cv2.imwrite(tmp, img)
                else:
                    raise Exception('Failed to capture image of {}, turning off camera.'.format(instrument))
            except Exception as exception:
                # turn off the camera                
                self.cameras[cam_id].release()
                # send warning to slack
                message = '{}\nAwaiting user input to continue or cancel pipeline.'.format(exception)
                self.slackbot.alert_msg(message)
                print('!! WARNING: {} !!'.format(exception))

                # await user y/n input to terminate or continue robot pipeline
                usr_input = input('Continue anyway without camera validation?\n')
                if usr_input[0].lower() == 'n':
                    kill_pipeline = True

                return False, kill_pipeline
            
        return True, kill_pipeline

    def close(self):
        # turn off all cameras
        for camera in self.cameras:
            camera.release()

if __name__ == '__main__':
    rv = RobotVision()
    rv.capture_image('dispenser')
