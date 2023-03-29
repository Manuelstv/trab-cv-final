import shutil
import os
import cv2

path = "/home/msnuel/trab-final-cv/animals/rhino2"
dirs = os.listdir( path )

import numpy as np

for file in dirs:
    if file.endswith('jpg'):
        
        annot_file = f'{path}/{file[:-3]}txt'
        
        im = cv2.imread(f'{path}/{file}')
                
        im_h, im_w = im.shape[0], im.shape[1]
        img_shape = (im_h,im_w)
        
        reshaped_img_shape = (256, 256)
        
        scale = np.flipud(np.divide(reshaped_img_shape, img_shape))	     
               
        im_resized = cv2.resize(im, reshaped_img_shape, interpolation = cv2.INTER_AREA)
        
        f = open(annot_file,"r")
        lines = f.read()
        
        print(annot_file)
        labels, x, y, w, h = lines.split()
        
        f = open(annot_file,"w")
        
        top_left_corner = ((float(x)-float(w)/2)*im_w/256, (float(y)-float(h)/2)*im_h/256)
        bottom_right_corner = ((float(x)+float(w)/2)*im_w/256, (float(y)+float(h)/2)*im_h/256)
        
        
        new_top_left_corner = np.multiply(top_left_corner, scale )
        new_bottom_right_corner = np.multiply(bottom_right_corner, scale )
        
        f.write(f'{labels} {new_top_left_corner[0]} {new_top_left_corner[1]} {new_bottom_right_corner[0]} {new_bottom_right_corner[1]}')
        cv2.imwrite(f'{path}/{file}', im_resized)
    
