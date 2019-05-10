import numpy as np
import cv2
from matplotlib import pyplot as plt
import imutils
import argparse
import math

MIN_MATCH_COUNT = 2
img100 =cv2.imread('yuurei.ppm')                   #read
e1 = cv2.getTickCount()                          # time
img10 =cv2.cvtColor(img100, cv2.COLOR_BGR2RGB)   #colorful picture
img1 = cv2.cvtColor(img100, cv2.COLOR_BGR2GRAY)  # gray picture
img200 =cv2.imread('class3_b1_n100_2.ppm')
img20 =cv2.cvtColor(img200, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img200, cv2.COLOR_BGR2GRAY)
#plt.imshow(img2, 'gray'),plt.show()
#img1 = cv2.imread('usamimi2.ppm',0)
#img2 = cv2.imread('class2_b2_n50_1.ppm',0)

sift = cv2.xfeatures2d.SIFT_create()            # sift

kp1, des1 = sift.detectAndCompute(img10,None)   # keypiont
kp2, des2 = sift.detectAndCompute(img20,None)

FLANN_INDEX_KDTREE = 0                          #kd树        knn match
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)


# store all the good matches as per Lowe's ratio test.

good = []                                        #good matches
for m,n in matches:
    if m.distance < 0.7543*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)# draw
    dst = cv2.perspectiveTransform(pts,M)                                # matche tmeplate
    pts1 = np.float32([ [0,h-1], [w - 1, 0], ]).reshape(-1, 1, 2)
    dst1 = cv2.perspectiveTransform(pts1, M)
    #h1, w1 = img2.shape
    #img2002 = np.zeros((h1, w1, 3), np.uint8)
    #img2000 = cv2.polylines(img2002,[np.int32(dst)],True,(255,255,255) ,3, cv2.LINE_AA)
    #img2000 =cv2.fillPoly(img2001, [np.int32(dst)],(255,255,255))
    #print(dst1)
    img20 =  cv2.polylines(img20, [np.int32(dst)], True, (0, 0, 0), 15, cv2.LINE_AA)
    cv2.imwrite('result.ppm', img20)
    #plt.imshow(img2000, 'gray'), plt.show()
else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img10,kp1,img20,kp2,good,None,**draw_params)

   #plt.imshow(img3, 'gray'),plt.show()

#HSV = cv2.cvtColor(img2000, cv2.COLOR_BGR2HSV)
#H, S, V = cv2.split(HSV)
#LowerRed = np.array([100, 100, 50])
#UpperRed = np.array([130, 255, 255])
#mask = cv2.inRange(HSV, LowerRed, UpperRed)
#RedThings = cv2.bitwise_and(img2000, img2000, mask=mask)
#cv2.imwrite('red.ppm',RedThings)
#image = cv2.imread("red.ppm")
vector1 = np.array(dst1[0])    #caculate
vector2 = np.array(dst1[1])
op1=np.sqrt(np.sum(np.square(vector1-vector2)))  #scaling
vector3 = np.array([0,h])
vector4 = np.array([w,0])
op2=np.sqrt(np.sum(np.square(vector3-vector4)))
scaling = op1/op2
print('scaling = ',scaling)
center = (vector1+vector2)/2
print('center = ',center)

x = vector2-vector1          #angle
y = vector3-vector4
cos_angle = x.dot(y)/(op1*op2)
angle = np.arccos(cos_angle)
sin_angle = -math.cos(np.pi/2-angle) # coa -> sin
angle2 = sin_angle*360/2/np.pi
print('angle =',angle2)

e2 = cv2.getTickCount()                 # time
time = (e2-e1)/cv2.getTickFrequency()
print('time = ', time)
plt.imshow(img3, 'gray'),plt.show()
