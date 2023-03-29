from ast import Assign
from collections import OrderedDict
#from pascal_voc_writer import Writer as wr

import sys
import os
import cv2
import numpy as np
import pandas as pd


path = r'/home/msnuel/trab-final-cv/animals/train/image_0.jpg'

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

df = pd.read_csv(r'/home/msnuel/trab-final-cv/animals/train/image_0.txt', sep =' ', header=None)

df.columns = ['class','x', 'y','h','w']

new_df = pd.DataFrame()
new_df['class'] = df['class']

new_df['a'] = (df['x'])*256 #*img_width
new_df['b'] = (df['y'])*256     #*img_height

new_df['c'] = df['h']*256 #+df['w']/2)*img_width
new_df['d'] = df['w']*256 #+df['h']/2)*img_height

#print(new_df)
#new_df = new_df.reset_index()
#print(new_df) 

for index, row in new_df.iterrows():
	# Using cv2.rectangle() method
	# Start coordinate, here (5, 5)
	# represents the top left corner of rectangle
	#print(row['a'], row['b'])
	start_point = (int(row['a']), int(row['b']))
	# Ending coordinate, here (220, 220)
	# represents the bottom right corner of rectangle
	end_point = (int(row['c']), int(row['d']))
	print(end_point)
	# Draw a rectangle with blue line borders of thickness of 2 px
	image = cv2.rectangle(image, start_point, end_point, color, thickness)
# Displaying the image 
#cv2.imshow(window_name, image)
#cv2.waitKey(0)

cv2.imwrite('outfile.jpg', image)

