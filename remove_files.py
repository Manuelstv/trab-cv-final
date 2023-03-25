import shutil
import os

path = "/home/msnuel/trab-final-cv/traffic_sign/new_labels"
dirs = os.listdir( path )

# This would print all the files and directories
for file in dirs:
    if file.endswith('png'):
        if os.path.exists(f'/home/msnuel/trab-final-cv/traffic_sign/new_labels/{file[:-3]}txt'):
            pass
        else:
            os.remove(f'/home/msnuel/trab-final-cv/traffic_sign/new_labels/{file}')
        
    
