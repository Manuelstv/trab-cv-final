import os
path = "/home/msnuel/trab-final-cv/cross_val_sph/dataset_fold_2/val"
value = 0
for file in os.listdir(path):
    new_filename = f'{path}/img_{value}.jpg'
    new_filename2 = f'{path}/img_{value}.txt'
    #print(file)
    if file.endswith('jpg'):
        print(file)
        if os.path.exists(new_filename):
            #print(new_filename)
            pass
        else:
            print(file)
            os.rename(f'{path}/{file}', new_filename)
            os.rename(f'{path}/{file[:-3]}txt', new_filename2)
            value += 1
