# takepic.py
#
# Help script for collecting data of one instument for DNN training
#
# Run the program by being in the same directory as this file with:
# $ python takepic.py folder/to/save/images/ instrument
#
# instument can be "CyclerX", "Dispenser", "Centrifuge"
#
# Change the number of picutes taken by changing the @num_pics variable
# Update coordinates if cameras have been moved by changing variables:
# @cen_x_dispenser
# @cen_y_dispenser
# @cen_x_cycler
# @cen_y_cycler
# @cen_x_centrifuge
# @cen_y_centrifuge
#
# If the cameras are changed, update the ID values in variables:
# @cycler_camera1
# @cycler_camera2
# @dispenser_camera 
# @centrifuge_camera 
#
# Author: Arnold Teo
# Version: 2023-04-21
import cv2, time, sys
from datetime import datetime
import asyncio
import winrt.windows.devices.enumeration as windows_devices

cen_x_dispenser = 1045
cen_y_dispenser = 604
cen_x_cycler = [1545, 1220, 855, 1395]
cen_y_cycler = [990,  980,  970, 1055]
cen_x_centrifuge = 320
cen_y_centrifuge = 240

num_pics = 5

cycler_camera1_id = '\\\\?\\USB#VID_1BCF&PID_28C4&MI_00#7&17f36d47&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\\GLOBAL'
cycler_camera2_id = '\\\\?\\USB#VID_1BCF&PID_28C4&MI_00#7&13d8f827&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\\GLOBAL'
dispenser_camera_id = '\\\\?\\USB#VID_046D&PID_08E5&MI_00#6&114c98e7&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\\GLOBAL'
centrifuge_camera_id = '\\\\?\\USB#VID_8086&PID_0B07&MI_03#7&4269cf7&0&0003#{e5323777-f976-4f5b-9b55-b94699c46e44}\\GLOBAL'

save_folder = sys.argv[1]
instrument = sys.argv[2]

img_height = 224
img_width = 224

async def get_camera_info():
    # from https://stackoverflow.com/questions/73946689/get-usb-camera-id-with-open-cv2-on-windows-10
    return await windows_devices.DeviceInformation.find_all_async(4)

connected_cameras = asyncio.run(get_camera_info())
cam_names = [camera.name for camera in connected_cameras]
cam_ids = [camera.id for camera in connected_cameras]

print(cam_names)
print(cam_ids)

# Should look something like this, the arrays have corresponding indexes
#['Intel(R) RealSense(TM) Depth Camera 435 with RGB Module Depth', 'Sandberg Pro Elite 4K UHD', 'Intel(R) RealSense(TM) Depth Camera 435 with RGB Module RGB', 'HD Pro Webcam C920', 'Sandberg Pro Elite 4K UHD']
#['\\\\?\\USB#VID_8086&PID_0B07&MI_00#7&4269cf7&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\\GLOBAL', '\\\\?\\USB#VID_1BCF&PID_28C4&MI_00#7&13d8f827&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\\GLOBAL', '\\\\?\\USB#VID_8086&PID_0B07&MI_03#7&4269cf7&0&0003#{e5323777-f976-4f5b-9b55-b94699c46e44}\\GLOBAL', '\\\\?\\USB#VID_046D&PID_08E5&MI_00#6&114c98e7&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\\GLOBAL', '\\\\?\\USB#VID_1BCF&PID_28C4&MI_00#7&17f36d47&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\\GLOBAL']

cam_id = ''
plate_cen_x = 0
plate_cen_y = 0

frame_h = 0
frame_w = 0
focus_depth = 0

if instrument.lower() == 'dispenser': 
    cam_id = dispenser_camera_id
    plate_cen_x = cen_x_dispenser
    plate_cen_y = cen_y_dispenser
    frame_h = 1080
    frame_w = 1920
    focus_depth = 15
elif instrument[:6].lower() == 'cycler':
    cycler_num = instrument[6:]
    if (cycler_num in ['1', '2']):
        cam_id = cycler_camera1_id
    else:
        cam_id = cycler_camera2_id
    plate_cen_x = cen_x_cycler[int(cycler_num)-1]
    plate_cen_y = cen_y_cycler[int(cycler_num)-1]
    frame_h = 1536
    frame_w = 2048
    focus_depth = 10
elif instrument.lower() == 'centrifuge': 
    cam_id = centrifuge_camera_id
    plate_cen_x = cen_x_centrifuge
    plate_cen_y = cen_y_centrifuge
    frame_h = 480
    frame_w = 640
    focus_depth = 5

camID = 99

if cam_id in cam_ids:
    camID = cam_ids.index(cam_id)

# capture image with chosen camera
cap = cv2.VideoCapture(camID)

# set frame dimensions
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

# set autofocus off
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  

# set focus depth
cap.set(cv2.CAP_PROP_FOCUS, focus_depth)

for i in range(num_pics):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    filename = "{}{}.png".format(save_folder, dt_string)

    # capture frame
    ret, frame = cap.read()

    # save cropped frame
    if ret:
        img = frame[(plate_cen_y-img_height//2):(plate_cen_y+img_height//2), (plate_cen_x-img_width//2):(plate_cen_x+img_width//2), :]
        cv2.imwrite(filename, img)

    time.sleep(1)

cap.release()
