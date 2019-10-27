import csv
import numpy as np
from PIL import Image
# from pathlib import os
size = 64, 64

def loadData(labels_path, images_path):
    labels_out = []
    images_out = []
    with open(labels_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        #del(csv_reader[0])
        x = 0
        for row in csv_reader:
            # if x != 0 and row[2] != "validation":
            if x != 0:
                #path = os.path.join(images_path, row[2], row[0])
                path = images_path + "/" + row[2] + "/" + row[0]
                img = Image.open(path)
                img = img.convert("L")
                img = img.resize(size)
                img = np.array(img)
                img2 = []
                for i in img:
                    i2 = i / 255
                    img2.append(i2)


                if row[2] == "validation":
                    row[2] = 0
                elif row[2] == "house":
                    row[2] = 1
                elif row[2] == "dinning_room":
                    row[2] = 2
                elif row[2] == "kitchen":
                    row[2] = 3
                elif row[2] == "bathroom":
                    row[2] = 4
                elif row[2] == "living_room":
                    row[2] = 5
                elif row[2] == "bedroom":
                    row[2] = 6     
                # del(row[2])
                del(row[0])
                # print (img)
                labels_out.append(row)
                images_out.append(img2)
                x += 1
                print(x, " - OK")
            else:
                x += 1
                print(x, " - 0 / VALIDATION")
            
            
    
    labels_out2 = np.asarray(labels_out)
    images_out2 = np.asarray(images_out)
    return labels_out2, images_out2

# a, b = loadData('bin/labels.csv', 'bin/images')
# print(a)
# print(b)
