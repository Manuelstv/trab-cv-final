import shutil
import os

path = "/home/msnuel/trab-final-cv/animals/elephant"
dirs = os.listdir( path )

# This would print all the files and directories
for file in dirs:
    if file.endswith('txt'):
        with open(f'{path}/{file}') as f:
            line_count = 0
            for line in f:
                line_count += 1
            if (line_count == 1):
                shutil.copyfile(f'{path}/{file}', f'/home/msnuel/trab-final-cv/animals/elephant2/{file}')
                shutil.copyfile(f'{path}/{file[:-3]}jpg', f'/home/msnuel/trab-final-cv/animals/elephant2/{file[:-3]}jpg')
        
    
