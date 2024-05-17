from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import numpy as np
def loadData(filename):
    print('start to read data')
    dataArr =[]
    labelArr=[]
    fr=open(filename,'r')
    for line in tqdm(fr.readlines()):
        curLine=line.strip().split(',')
        labelArr.append(curLine[0])
        dataArr.append([int(num)/255 for num in curLine[1:]])
    return dataArr,labelArr


trainData,trainLabel=loadData('./mnist_train.csv')
testData,testLabel=loadData('./mnist_test.csv')
print('start to classify')
clf_sk = KNeighborsClassifier(n_neighbors=2)
clf_sk.fit(trainData,trainLabel)
print('accurate is:',clf_sk.score(testData, testLabel))

