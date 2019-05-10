import cv2
import os
from xlwt import *

file = Workbook(encoding = 'utf-8')
table = file.add_sheet('negative.xls')
path = "negative1"
filename = os.listdir(path)
#print(filename)
#for file in os.listdir(r"./Positive"):              #listdir的参数是文件夹的路径
     #print (filename)
j = 0
for a in filename:
    newDir = os.path.join(path,a)
    img = cv2.imread(newDir)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(50)
    kp, des = sift.detectAndCompute(img_gray, None)
    for i in range(0,20):
        tu = kp[i].pt  #（提取坐标） pt指的是元组 tuple(x,y)
        #a = kp[i].angle
        #b = kp[i].size
        #c = kp[i].response
        #d = kp[i].octave
        #print(tu[0]," "  , tu[1], " ",1)
        table.write(20*j+i, 0, tu[0])
        table.write(20*j+i, 1, tu[1])
        table.write(20*j+i, 2, -1)
        #table.write(20*j+i*2+1, 0, a)
        #table.write(20*j+i*2+1, 1, b)
        #table.write(20*j+i*2+1, 2, -1)

    j=j+1
    print(j)
    print(newDir)
    if j >500:
        break
file.save('negative.xls')