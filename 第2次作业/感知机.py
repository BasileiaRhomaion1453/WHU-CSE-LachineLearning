import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

loss=[]
def loadData(filename):
    print('start to read data')
    dataArr =[]
    labelArr=[]
    fr=open(filename,'r')
    for line in tqdm(fr.readlines()):
        curLine=line.strip().split(',')
        if int(curLine[0])>=5:
            labelArr.append(1)
        else:
            labelArr.append(-1)
        dataArr.append([int(num)/255 for num in curLine[1:]])
    return dataArr,labelArr

def pretrain(dataArr,labelArr,iter=50):
    print('star to trans')
    dataMat=np.mat(dataArr)
    labelMat=np.mat(labelArr).T
    m,n=np.shape(dataMat)
    w=np.zeros((1,np.shape(dataMat)[1]))
    b=0
    h=0.0001
    for k in range(iter):
        for i in range (m):
            xi=dataMat[i]
            yi=labelMat[i]
            if (yi*(w*xi.T+b)<=0):
                w=w+h*yi*xi
                b=b+h*yi
                print('%d:%d training' %(k,iter))
        loss.append(pretest(testData,testLabel,w,b))            
    return w,b

def pretest(dataArr,labelArr,w,b):
    print('start to test')
    dataMat=np.mat(dataArr)
    labelMat=np.mat(labelArr).T
    m,n=np.shape(dataMat)
    errorCnt=0
    for j in range(m):
        xi=dataMat[j]
        yi=labelMat[j]
        result=yi*(w*xi.T+b)
        if result<=0:
            errorCnt+=1
            loss=(errorCnt/m)
    return loss 

if __name__=='__main__':
    trainData,trainLabel=loadData('./mnist_train.csv')
    testData,testLabel=loadData('./mnist_test.csv')
    w,b=pretrain(trainData,trainLabel,iter=100)
    #loss=pretest(testData,testLabel,w,b)
    print('accuracy rate is:',loss)
    x_axis_data = range(100)
    plt.plot(x_axis_data,loss, alpha=0.5)
    plt.show()

