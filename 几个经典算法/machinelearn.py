from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import math
import random
import time
from sklearn.metrics import classification_report
def loadData(fileName):
    dataArr = []; labelArr = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        #在放入的同时将原先字符串形式的数据转换为0-1的浮点型
        dataArr.append([int(num) / 255 for num in curLine[1:]])
        if int(curLine[0]) == 0:
            labelArr.append(1)
        else:
            labelArr.append(-1)
    return dataArr, labelArr

def ID3loadData(filename):
    # 存放数据以及标记
    dataArr = []
    labelArr = []
    # 读取文件
    fr = open(filename)
    # 遍历文件
    for line in fr.readlines():
        # strip:去除首尾部分的空格和回车
        curline = line.strip().split(",")
        # 数据二值化操作，减小运算量
        dataArr.append([int(int(num) > 128) for num in curline[1:]])
        # 添加标记
        labelArr.append(int(curline[0]))
        # 返回数据集和标记
    return dataArr, labelArr

class ID3():
    def MajorClass(self,labelArr):
        '''
    找到当前标签集中占数目最大的标签
    :param labelArr: 标签集
    :return: 最大的标签
        '''
    # 建立字典，统计不同类别标签的数量
        classDict = {}
    # 遍历所有标签
        for i in range(len(labelArr)):
            if labelArr[i] in classDict.keys():
                classDict[labelArr[i]] += 1
            else:
                classDict[labelArr[i]] = 1
    # 对不同类别标签的统计情况进行降序排序
        classSort = sorted(classDict.items(), key=lambda x: x[1], reverse=True)
    # 返回最大一项的标签，即占数目最多的标签
        return classSort[0][0]

    def Cal_HD(self,trainLabelArr):
        '''
    计算数据集D的经验熵，参考公式经验熵的计算
    :param trainLabelArr:当前数据集的标签集
    :return: 经验熵
    '''
        HD = 0
    # 统计该分支的标签情况
    # set()删除重复数据
        trainLabelSet = set([label for label in trainLabelArr])
    # 遍历每一个出现过的标签
        for i in trainLabelSet:
            p = trainLabelArr[trainLabelArr == i].size / trainLabelArr.size
        # 对经验熵的每一项进行累加求和
            HD += -1 * p * np.log2(p)

    # 返回经验熵
        return HD
    def Cal_HDA(self,trainDataArr_DevFeature, trainLabelArr):
        '''
    计算经验条件熵
    :param trainDataArr_DevFeature:切割后只有feature那列数据的数组
    :param trainLabelArr: 标签集数组
    :return: 条件经验熵
    '''
    # 初始为0
        HDA = 0
    # 拿到当前指定feature中的可取值的范围
        trainDataSet = set([label for label in trainDataArr_DevFeature])

    # 对于每一个特征取值遍历计算条件经验熵的每一项
    # trainLabelArr[trainDataArr_DevFeature == i]表示特征值为i的样本集对应的标签集
        for i in trainDataSet:
            HDA += trainDataArr_DevFeature[trainDataArr_DevFeature == i].size / trainDataArr_DevFeature.size * self.Cal_HD(
                trainLabelArr[trainDataArr_DevFeature == i])
    # 返回得出的条件经验熵
        return HDA
    def Cal_BestFeature(self,trainDataList, trainLabelList):
        '''
    计算信息增益最大的特征
    :param trainDataList: 当前数据集
    :param trainLabelList: 当前标签集
    :return: 信息增益最大的特征及最大信息增益值
    '''
    # 列表转为数组格式
        trainDataArr = np.array(trainDataList)
        trainLabelArr = np.array(trainLabelList)

    # 获取当前的特征个数
    # 获取trainDataArr的列数
        featureNum = trainDataArr.shape[1]

    # 初始化最大信息熵G(D|A)
        max_GDA = -1
    # 初始化最大信息增益的特征索引
        max_Feature = -1

    # 计算数据集的经验熵
        HD = self.Cal_HD(trainLabelArr)
    # 对每一个特征进行遍历计算
        for feature in range(featureNum):
        # .flat：flat返回的是一个迭代器，可以用for访问数组每一个元素
            trainDataArr_DevideByFeature = np.array(trainDataArr[:, feature].flat)

        # 计算信息增益G(D|A) = H(D) - H(D|A)
            GDA = HD - self.Cal_HDA(trainDataArr_DevideByFeature, trainLabelArr)

        # 不断更新最大的信息增益以及对应的特征
            if GDA > max_GDA:
                max_GDA = GDA
                max_Feature = feature
        return max_Feature, max_GDA
    def GetSubDataArr(self,trainDataArr, trainLabelArr, A, a):
        '''
    待更新的子数据集和标签集
    :param trainDataArr: 待更新的数据集
    :param trainLabelArr: 待更新的标签集
    :param A: 待去除的选定特征
    :param a: 当data[A]==a时，该行样本保留
    :return: 新的数据集和标签集
    '''
    # 返回的数据集，标签集
        retDataArr, retLabelArr = [], []

    # 对当前数据集的每一个样本进行遍历
        for i in range(len(trainDataArr)):
        # 如果当前样本的特征为指定特征值a
            if trainDataArr[i][A] == a:
            # 那么将该样本的第A个特征切割掉，放入返回的数据集中
                retDataArr.append(trainDataArr[i][0:A] + trainDataArr[i][A + 1:])
            # 将该样本的标签放入新的标签集中
                retLabelArr.append(trainLabelArr[i])

    # 返回新的数据集和标签集
        return retDataArr, retLabelArr
    def CreateTree(self,*dataSet):
        '''
    递归创建决策树
    :param dataSet:(trainDataList， trainLabelList) <<-- 元祖形式
    :return:新的子节点或该叶子节点的值
    '''
        Epsilon = 0.1

        trainDataList = dataSet[0][0]
        trainLabelList = dataSet[0][1]

    # 创建子节点，打印当前的特征个数，以及当前的样本个数
        #print("start a new lode,当前的特征个数为:%d，样本个数为:%d" % (len(trainDataList[0]), len(trainLabelList)))

    # 将标签放入一个字典中，当前样本有多少类，在字典中就会有多少项
        classDict = {i for i in trainLabelList}
        if len(classDict) == 1:
        # 因为所有样本都是一致的，在标签集中随便拿一个标签返回都行
            return trainLabelList[0]

    # 当特征个数为空时，返回占最多数的类别标签
        if len(trainDataList[0]) == 0:
            return self.MajorClass(trainLabelList)

    # 否则，计算每个特征的信息增益，选择信息增益最大的特征Ag
        Ag, max_GDA = self.Cal_BestFeature(trainDataList, trainLabelList)

    # Ag的信息增益小于阈值Epsilon，则置T为单节点树，并将D中实例数最大的类Ck，作为该节点的类，返回T。
        if max_GDA < Epsilon:
            return self.MajorClass(trainLabelList)

    # 否则，对Ag的每一个可能值ai，依据Ag=ai将数据集分割为若干个非空子集Di，将Di中实例数最大的类作为标记，
    # 构建子节点，由节点及其子节点构成树T，返回T。
        treeDict = {Ag: {}}
        treeDict[Ag][0] = self.CreateTree(self.GetSubDataArr(trainDataList, trainLabelList, Ag, 0))
        treeDict[Ag][1] = self.CreateTree(self.GetSubDataArr(trainDataList, trainLabelList, Ag, 1))
        return treeDict
    def Predict(self,testDataList,tree):

    # 死循环，直到找到一个有效地分类
        while True:
        # 以tree={73: {0: {74:6}}}为例,key=73，value={0: {74:6}}
            (key, value), = tree.items()
        # 如果当前的value是字典，说明还需要遍历下去
            if type(tree[key]).__name__ == 'dict':
            # 获取目前所在节点的feature值，需要在样本中删除该feature
            # 因为在创建树的过程中，feature的索引值永远是对于当时剩余的feature来设置的
            # 所以需要不断地删除已经用掉的特征，保证索引相对位置的一致性
                dataVal = testDataList[key]
                del testDataList[key]
            # 将tree更新为其子节点的字典
                tree = value[dataVal]
                if type(tree).__name__ == 'int':
                # 如果当前节点的子节点的值是int，就直接返回该int值
                # 以{403: {0: 7, 1: {297:7}}为例，dataVal=0，则当前节点的子节点的值是7，为int型
                # 返回该节点值，也就是分类值
                    return tree
            else:
            # 如果当前value不是字典，那就返回分类值
            # 以{297:7}为例，key=297,value=7，则直接返回7
                return value
    def Model_Test(self,testDataList, testLabelList,tree):
        '''
    测试准确率
    :param testDataList:待测试数据集
    :param testLabelList: 待测试标签集
    :param tree: 训练集生成的树
    :return: 准确率
    '''
    # 错误次数计数
        errorCnt = 0
    # 遍历测试集中每一个测试样本
        for i in range(len(testDataList)):
        # 判断预测与标签中结果是否一致
            if testLabelList[i] != self.Predict(testDataList[i],tree):
                errorCnt += 1
    # 返回准确率
        print('accurate is:',1 - errorCnt / len(testDataList),end=' ')

class MySVM:

    def __init__(self,sigma = 10, C = 200, toler = 0.001):
        '''
        :param sigma: 高斯核中分母的σ
        :param C:软间隔中的惩罚参数
        :param toler:松弛变量
        '''
        self.trainDataMat = None        #训练数据集
        self.trainLabelMat = None       #训练标签集，为了方便后续运算提前做了转置，变为列向量
        self.m = None
        self.n = None                   #m：训练集数量    n：样本特征数目
        self.sigma = sigma                              #高斯核分母中的σ
        self.C = C                                      #惩罚参数
        self.toler = toler                              #松弛变量
        self.k = None                      #核函数（初始化时提前计算）
        self.b = 0                                      #SVM中的偏置b
        self.alpha = None   # α 长度为训练集数目
        self.E = None    #SMO运算过程中的Ei
        self.supportVecIndex = []                       #支持向量集

    def calcKernel(self):
        '''
        :return: 高斯核矩阵
        '''
        #初始化高斯核结果矩阵 大小 = 训练集长度m * 训练集长度m
        #k[i][j] = Xi * Xj
        k = [[0 for i in range(self.m)] for j in range(self.m)]

        #大循环遍历Xi，Xi为式7.90中的x
        for i in range(self.m):
            X = self.trainDataMat[i, :]
            #小循环遍历Xj，Xj为式7.90中的Z
            # 由于 Xi * Xj 等于 Xj * Xi，一次计算得到的结果可以
            # 同时放在k[i][j]和k[j][i]中，这样一个矩阵只需要计算一半即可
            #所以小循环直接从i开始
            for j in range(i, self.m):
                #获得Z
                Z = self.trainDataMat[j, :]
                #先计算||X - Z||^2
                result = (X - Z) * (X - Z).T
                #分子除以分母后去指数，得到的即为高斯核结果
                result = np.exp(-1 * result / (2 * self.sigma**2))
                #将Xi*Xj的结果存放入k[i][j]和k[j][i]中
                k[i][j] = result
                k[j][i] = result
        #返回高斯核矩阵
        return k

    def isSatisfyKKT(self, i):
        '''
        查看第i个α是否满足KKT条件
        :param i:α的下标
        '''
        gxi =self.calc_gxi(i)
        yi = self.trainLabelMat[i]
        if (math.fabs(self.alpha[i]) < self.toler) and (yi * gxi >= 1):
            return True
        elif (math.fabs(self.alpha[i] - self.C) < self.toler) and (yi * gxi <= 1):
            return True
        elif (self.alpha[i] > -self.toler) and (self.alpha[i] < (self.C + self.toler)) \
                and (math.fabs(yi * gxi - 1) < self.toler):
            return True

        return False

    def calc_gxi(self, i):
        '''
        计算g(xi)
        :param i:x的下标
        :return: g(xi)的值
        '''
        gxi = 0
        #index获得非零α的下标，并做成列表形式方便后续遍历
        index = [i for i, alpha in enumerate(self.alpha) if alpha != 0]
        #遍历每一个非零α，i为非零α的下标
        for j in index:
            #计算g(xi)
            gxi += self.alpha[j] * self.trainLabelMat[j] * self.k[j][i]
        #求和结束后再单独加上偏置b
        gxi += self.b
        #返回
        return gxi

    def calcEi(self, i):
        '''
        计算Ei
        :param i: E的下标
        :return:
        '''
        #计算g(xi)
        gxi = self.calc_gxi(i)
        #Ei = g(xi) - yi,直接将结果作为Ei返回
        return gxi - self.trainLabelMat[i]

    def getAlphaJ(self, E1, i):
        '''
        SMO中选择第二个变量
        :param E1: 第一个变量的E1
        :param i: 第一个变量α的下标
        :return: E2，α2的下标
        '''
        #初始化E2
        E2 = 0
        #初始化|E1-E2|为-1
        maxE1_E2 = -1
        #初始化第二个变量的下标
        maxIndex = -1
        #获得Ei非0的对应索引组成的列表，列表内容为非0Ei的下标i
        nozeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]
        #对每个非零Ei的下标i进行遍历
        for j in nozeroE:
            #计算E2
            E2_tmp = self.calcEi(j)
            #如果|E1-E2|大于目前最大值
            if math.fabs(E1 - E2_tmp) > maxE1_E2:
                #更新最大值
                maxE1_E2 = math.fabs(E1 - E2_tmp)
                #更新最大值E2
                E2 = E2_tmp
                #更新最大值E2的索引j
                maxIndex = j
        #如果列表中没有非0元素了（对应程序最开始运行时的情况）
        if maxIndex == -1:
            maxIndex = i
            while maxIndex == i:
                #获得随机数，如果随机数与第一个变量的下标i一致则重新随机
                maxIndex = int(random.uniform(0, self.m))
            #获得E2
            E2 = self.calcEi(maxIndex)
        #返回第二个变量的E2值以及其索引
        return E2, maxIndex

    def fit(self,trainData,trainLabel, iter = 100):
        self.trainDataMat = np.mat(trainData)       #训练数据集
        self.trainLabelMat = np.mat(trainLabel).T   #训练标签集，为了方便后续运算提前做了转置，变为列向量
        self.m, self.n = np.shape(self.trainDataMat)    
        self.alpha = [0] * self.trainDataMat.shape[0]
        self.E = [0 * self.trainLabelMat[i, 0] for i in range(self.trainLabelMat.shape[0])]
        self.k = self.calcKernel() 
        #iterStep：迭代次数，超过设置次数还未收敛则强制停止
        #parameterChanged：单次迭代中有参数改变则增加1
        iterStep = 0; parameterChanged = 1
        #如果没有达到限制的迭代次数以及上次迭代中有参数改变则继续迭代
        #parameterChanged==0时表示上次迭代没有参数改变，如果遍历了一遍都没有参数改变，说明
        #达到了收敛状态，可以停止了
        while (iterStep < iter) and (parameterChanged > 0):
            #打印当前迭代轮数
            #print('iter:%d:%d'%( iterStep, iter))
            #迭代步数加1
            iterStep += 1
            #新的一轮将参数改变标志位重新置0
            parameterChanged = 0
            #大循环遍历所有样本，用于找SMO中第一个变量
            for i in range(self.m):
                #查看第一个遍历是否满足KKT条件，如果不满足则作为SMO中第一个变量从而进行优化
                if self.isSatisfyKKT(i) == False:
                    #如果下标为i的α不满足KKT条件，则进行优化
                    #第一个变量α的下标i已经确定，接下来按照“7.4.2 变量的选择方法”第二步
                    #选择变量2。由于变量2的选择中涉及到|E1 - E2|，因此先计算E1
                    E1 = self.calcEi(i)
                    #选择第2个变量
                    E2, j = self.getAlphaJ(E1, i)
                    #参考“7.4.1两个变量二次规划的求解方法” P126 下半部分
                    #获得两个变量的标签
                    y1 = self.trainLabelMat[i]
                    y2 = self.trainLabelMat[j]
                    #复制α值作为old值
                    alphaOld_1 = self.alpha[i]
                    alphaOld_2 = self.alpha[j]
                    #依据标签是否一致来生成不同的L和H
                    if y1 != y2:
                        L = max(0, alphaOld_2 - alphaOld_1)
                        H = min(self.C, self.C + alphaOld_2 - alphaOld_1)
                    else:
                        L = max(0, alphaOld_2 + alphaOld_1 - self.C)
                        H = min(self.C, alphaOld_2 + alphaOld_1)
                    #如果两者相等，说明该变量无法再优化，直接跳到下一次循环
                    if L == H:   continue
                    #计算α的新值
                    #先获得几个k值，用来计算事7.106中的分母η
                    k11 = self.k[i][i]
                    k22 = self.k[j][j]
                    k21 = self.k[j][i]
                    k12 = self.k[i][j]
                    #依据式7.106更新α2，该α2还未经剪切
                    alphaNew_2 = alphaOld_2 + y2 * (E1 - E2) / (k11 + k22 - 2 * k12)
                    #剪切α2
                    if alphaNew_2 < L: alphaNew_2 = L
                    elif alphaNew_2 > H: alphaNew_2 = H
                    #更新α1，依据式7.109
                    alphaNew_1 = alphaOld_1 + y1 * y2 * (alphaOld_2 - alphaNew_2)
                    #依据“7.4.2 变量的选择方法”第三步式7.115和7.116计算b1和b2
                    b1New = -1 * E1 - y1 * k11 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k21 * (alphaNew_2 - alphaOld_2) + self.b
                    b2New = -1 * E2 - y1 * k12 * (alphaNew_1 - alphaOld_1) \
                            - y2 * k22 * (alphaNew_2 - alphaOld_2) + self.b
                    #依据α1和α2的值范围确定新b
                    if (alphaNew_1 > 0) and (alphaNew_1 < self.C):
                        bNew = b1New
                    elif (alphaNew_2 > 0) and (alphaNew_2 < self.C):
                        bNew = b2New
                    else:
                        bNew = (b1New + b2New) / 2
                    #将更新后的各类值写入，进行更新
                    self.alpha[i] = alphaNew_1
                    self.alpha[j] = alphaNew_2
                    self.b = bNew
                    self.E[i] = self.calcEi(i)
                    self.E[j] = self.calcEi(j)
                    #如果α2的改变量过于小，就认为该参数未改变，不增加parameterChanged值
                    if math.fabs(alphaNew_2 - alphaOld_2) >= 0.00001:
                        parameterChanged += 1
                #打印迭代轮数，i值，该迭代轮数修改α数目
                #print("iter: %d i:%d, pairs changed %d" % (iterStep, i, parameterChanged))
        #全部计算结束后，重新遍历一遍α，查找里面的支持向量
        for i in range(self.m):
            #如果α>0，说明是支持向量
            if self.alpha[i] > 0:
                #将支持向量的索引保存起来
                self.supportVecIndex.append(i)

    def calcSinglKernel(self, x1, x2):
        '''
        单独计算核函数
        :param x1:向量1
        :param x2: 向量2
        :return: 核函数结果
        '''
        result = (x1 - x2) * (x1 - x2).T
        result = np.exp(-1 * result / (2 * self.sigma ** 2))
        #返回结果
        return np.exp(result)

    def predict(self, x):
        result = 0
        for i in self.supportVecIndex:
            tmp = self.calcSinglKernel(self.trainDataMat[i, :], np.mat(x))
            result += self.alpha[i] * self.trainLabelMat[i] * tmp
        result += self.b
        return np.sign(result)

    def score(self, testDataList, testLabelList):
        errorCnt = 0
        for i in range(len(testDataList)):
            #print('test:%d:%d'%(i, len(testDataList)))
            result = self.predict(testDataList[i])
            if result != testLabelList[i]:
                errorCnt += 1
        return 1 - errorCnt / len(testDataList)

if __name__ == "__main__":
    print('start read TrainDataSet')
    trainData,trainLabel=loadData('./mnist_train.csv')
    trainDataList,trainLabelList=ID3loadData('./mnist_train.csv')
    print('start read TestDataSet')
    testData,testLabel=loadData('./mnist_test.csv')
    testDataList,testLabelList=ID3loadData('./mnist_test.csv')
    print('start to train, please wait...')
    clf=[SVC(),KNeighborsClassifier(),MySVM(),ID3()]
    for i in range(2):
            start = time.time()
            clf[i].fit(trainData,trainLabel)
            end=time.time()
            print('accurate is:',clf[i].score(testData, testLabel),'time:',end-start)
            #print(classification_report(testLabel,clf[i].predict(testData)))
    start=time.time()
    clf[2].fit(trainData[:6000],trainLabel[:6000])
    end=time.time()
    print('accurate is:',clf[2].score(testData[:1000], testLabel[:1000]),'time:',end-start)
    start=time.time()
    tree=clf[3].CreateTree((trainDataList, trainLabelList))
    end=time.time()
    clf[3].Model_Test(testDataList, testLabelList,tree)
    print('time:',end-start)
    