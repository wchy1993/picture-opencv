from numpy import *
import SVM


################## test svm #####################
## step 1: load data
#print("step 1: load data...")
dataSet = []
labels = []
fileIn = open('testdata.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split()
    dataSet.append([float(lineArr[0]), float(lineArr[1])])
    labels.append(float(lineArr[2]))

dataSet = mat(dataSet)
labels = mat(labels).T
train_x = dataSet[0:100, :]
train_y = labels[0:100, :]
#test_x = dataSet[61:101, :]
#test_y = labels[61:101, :]
dataSet1 = []
labels1 = []
fileIn = open('tesdata.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split()
    dataSet1.append([float(lineArr[0]), float(lineArr[1])])
    labels1.append(float(lineArr[2]))
dataSet1 = mat(dataSet1)
labels1 = mat(labels1).T
test_x = dataSet[0:100, :]
test_y = labels[0:100, :]

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
SVM.showSVM(svmClassifier)