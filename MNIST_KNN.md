# 手写数字识别

## KNN算法
```python
from numpy import *
import operator
from os import listdir

#K-近邻算法
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #通过欧氏距离计算每个样本和已知数据的距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):#选择距离最小的K个点
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)#对数据进行排序
    return sortedClassCount[0][0]
```


## 数据准备：将图片转化为测试向量

将32*32的二进制图片转化为1*1024的向量
```python
def img2vector(filename):
    returnVect = zeros((1,1024)) #创建一个1*1024空的Vector
    fr = open(filename)
    for i in range(32): #循环的读出前32行的数据
        lineStr = fr.readline()
        for j in range(32): #对于每一行的前32个数据，依次放入到1*1024的空的Vector中
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
```
```python
testVector=img2vector("testDigits/0_0.txt")
testVector[0,0:31]
```
```
array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.])
```

## 算法测试：使用K-近邻算法测试向量
```python
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')        #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print( "\nthe total error rate is: %f" % (errorCount/float(mTest)))

```


