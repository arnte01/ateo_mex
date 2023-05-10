# reposition_cycler_cam2.py
#
# View difference between images for retraining models
# Version: 2023-04-03
import numpy as np
import cv2

import asyncio
import winrt.windows.devices.enumeration as windows_devices

async def get_camera_info():
    #from: https://stackoverflow.com/questions/73946689/get-usb-camera-id-with-open-cv2-on-windows-10
    return await windows_devices.DeviceInformation.find_all_async(4)

sandberg_id2 = '\\\\?\\USB#VID_1BCF&PID_28C4&MI_00#7&13d8f827&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\\GLOBAL'

connected_cameras = asyncio.run(get_camera_info())
cam_ids = [camera.id for camera in connected_cameras]  

sandberg_index2 = 99
if sandberg_id2 in cam_ids:
    sandberg_index2 = cam_ids.index(sandberg_id2)


cap = cv2.VideoCapture(sandberg_index2)

# set frame dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
# set autofocus off
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  
# set focus depth
cap.set(cv2.CAP_PROP_FOCUS, 10)

# read in images
file_old = 'cycler2_reference.png'
img_old = cv2.imread(file_old)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        transp = 0.3
        disp_frame =  ((1-transp) * frame.astype(np.float) + transp * img_old.astype(np.float)).astype(np.uint8)
        disp_frame = cv2.resize(disp_frame, (1920, 1080))

        cv2.imshow('frame', disp_frame)

        # Exit if ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
