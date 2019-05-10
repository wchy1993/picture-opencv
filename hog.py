import cv2
import os
from xlwt import *
import numpy as np
#file = Workbook(encoding = 'utf-8')
#table = file.add_sheet('hog.xls')
path = "Negative1"
filename = os.listdir(path)
j = 0
for a in filename:
    newDir = os.path.join(path, a)
    img = cv2.imread(newDir)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor((64,128), (16,16), (8,8), (8,8), 9)
     #img = cv2.imread(f, -1)
     #img = cv2.resize(img, (64,128))
    descriptors = hog.compute(img)
    fileIn = open(a +".txt", "w")
    for i in range(0,3500):
        tu = np.array(descriptors)
        #print(newDir)
        #print(j," ",tu[i], " ", tu[i+1])
        print(tu[i], " ", tu[i+1], file = fileIn)

        #a = kp[i].angle
        #b = kp[i].size
        #c = kp[i].response
        #d = kp[i].octave
        #print(tu[0]," "  , tu[1], " ",1)
        #table.write(e, 0, hog1[e])
        #table.write(1500*e, 1, hog1[e+1500])
        #table.write(j, i, descriptors[1])
        #table.write(j, 2, -1)
    j=j+1
    #print(j)
    #print(newDir)
    if j >500:
        break


fileIn.close()

#file.save('hog.xls')


#print(descriptors[0])