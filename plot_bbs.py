from ast import Assign
from collections import OrderedDict
#from pascal_voc_writer import Writer as wr

import sys
import os
import cv2
import numpy as np
import pandas as pd


path = r'/home/msnuel/trab-final-cv/outfile.jpg'

# Reading an image in default mode
image = cv2.imread(path)

# Window name in which image is displayed
window_name = 'Image'

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

img_width = image.shape[1]
img_height = image.shape[0]

#df = pd.DataFrame()

df = pd.read_csv(r'/home/msnuel/trab-final-cv/animals/train/image_10.txt', sep =' ', header=None)

df.columns = ['class','x', 'y','h','w']

new_df = pd.DataFrame()
new_df['class'] = df['class']

new_df['x_min'] = (df['x'])*256 #*img_width
new_df['y_min'] = (df['y'])*256     #*img_height

new_df['x_max'] = df['h']*256 #+df['w']/2)*img_width
new_df['y_max'] = df['w']*256 #+df['h']/2)*img_height

#print(new_df)
#new_df = new_df.reset_index()
#print(new_df) 

for index, row in new_df.iterrows():
	# Using cv2.rectangle() method
	# Start coordinate, here (5, 5)
	# represents the top left corner of rectangle
	#print(row['a'], row['b'])
	start_point = (int(row['x_min']), int(row['y_min']))
	# Ending coordinate, here (220, 220)
	# represents the bottom right corner of rectangle
	end_point = (int(row['x_max']), int(row['y_max']))
	print(end_point)
	# Draw a rectangle with blue line borders of thickness of 2 px
	image = cv2.rectangle(image, start_point, end_point, color, thickness)
# Displaying the image 
#cv2.imshow(window_name, image)
#cv2.waitKey(0)

cv2.imwrite('outfile.jpg', image)

