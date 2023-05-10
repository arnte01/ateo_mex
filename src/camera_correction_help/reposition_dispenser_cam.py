# reposition_dispenser_cam.py
#
# View difference between images for retraining models
# Version: 2023-04-20
import numpy as np
import cv2

import asyncio
import winrt.windows.devices.enumeration as windows_devices

async def get_camera_info():
    #from: https://stackoverflow.com/questions/73946689/get-usb-camera-id-with-open-cv2-on-windows-10
    return await windows_devices.DeviceInformation.find_all_async(4)

logitech_name = 'HD Pro Webcam C920'

connected_cameras = asyncio.run(get_camera_info())
cam_names = [camera.name for camera in connected_cameras]  

logitech_index = 99
if logitech_name in cam_names:
    logitech_index = cam_names.index(logitech_name)

cap = cv2.VideoCapture(logitech_index)

# set frame dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# set autofocus off
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  
# set focus depth
cap.set(cv2.CAP_PROP_FOCUS, 15)

# capture frame
ret, frame = cap.read()

# read in reference image
file_old = 'dispenser_reference.png'
img_old = cv2.imread(file_old)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        transp = 0.3
        disp_frame =  ((1-transp) * frame.astype(np.float) + transp * img_old.astype(np.float)).astype(np.uint8)
        
        cv2.imshow('frame', disp_frame)

        # Exit if ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()