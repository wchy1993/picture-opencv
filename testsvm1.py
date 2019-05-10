from numpy import *
import SVM
import cv2
from matplotlib import pyplot as plt
################## test svm #####################
## step 1: load data
#print("step 1: load data...")
img = cv2.imread("Positive1/per00189.ppm")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create(50)
kp, des = sift.detectAndCompute(img_gray, None)
fileIn = open("out.txt","w")
for i in range(0, 20):
    tu = kp[i].pt  # （提取坐标） pt指的是元组 tuple(x,y)
    #a = kp[i].angle
    #b = kp[i].size
    print(tu[0]," ", tu[1]," ",1)
    print(tu[0], " ", tu[1], " ", 1, file = fileIn)
0

dataSet = []
labels = []
fileIn = open('pedestrian.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split()
    dataSet.append([float(lineArr[0]), float(lineArr[1])])
    labels.append(float(lineArr[2]))
dataSet = mat(dataSet)
labels = mat(labels).T
train_x = dataSet[0:199, :]
train_y = labels[0:199, :]
#test_x = dataSet[61:101, :]
#test_y = labels[61:101, :]

dataSet1 = []
labels1 = []

#fileIn = open("out.txt")
fileIn = open('positive.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split()
    dataSet1.append([float(lineArr[0]), float(lineArr[1])])
    labels1.append(float(lineArr[2]))
dataSet1 = mat(dataSet1)
labels1 = mat(labels1).T
test_x = dataSet[0:20, :]
test_y = labels[0:20, :]

dataSet = mat(dataSet)
labels = mat(labels).T

## step 2: training...
#print("step 2: training...")
C = 1.0
toler = 0.001
maxIter = 50
#def Gauss_kernel(x,z,sigma=1):
		#return np.exp(-np.sum((x-z)**2)/(2*sigma**2))
svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption=('linear', 0))

## step 3: testing
#print("step 3: testing...")
accuracy = SVM.testSVM(svmClassifier, test_x, test_y)

## step 4: show the result
#print("step 4: show the result...")
print('The classify accuracy is: %.3f%%' % (accuracy * 100))
if  accuracy> 0.5:
    print("this is pedestrain")
else:
    print("this is not pedestrain")
SVM.showSVM(svmClassifier)
plt.imshow(img), plt.show()