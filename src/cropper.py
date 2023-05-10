# cropper.py
# crops raw images to 224x224 images
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from PIL import Image
import os

img_height = 224
img_width = 224

# dispenser
#plate_cen_y = 604
#plate_cen_x = 1045

# centrifuge
plate_cen_x = 320
plate_cen_y = 240

# path to data folder
file_path = '../data/path/to/folder/'
sub_folders = ['train/', 'valid/']

for folder in sub_folders:
    directory_files = os.listdir(file_path+folder)
    multiple_images = [file for file in directory_files if file.endswith(('.jpg', '.png'))]

    #print(directory_files)
    #print(multiple_images)

    for filename in multiple_images:
        img = Image.open(file_path+folder+filename)
        img_crp = mpimg.pil_to_array(img)

        img_crp = img_crp[(plate_cen_y-img_height//2):(plate_cen_y+img_height//2), (plate_cen_x-img_width//2):(plate_cen_x+img_width//2), :]

        img_crp = Image.fromarray(img_crp)

        #plt.imshow(img_crp)
        #plt.show()

        img_crp.save(file_path+folder+'crp_'+filename)
