from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB#贝叶斯
from sklearn.neighbors import KNeighborsClassifier#k近邻
from sklearn.linear_model import Perceptron#感知机
from sklearn.svm import SVC#支持向量机
from sklearn.neural_network import MLPClassifier#多层感知机
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import classification_report
from tqdm import tqdm
def predata():
    train=fetch_20newsgroups(data_home=None, # 文件下载的路径
                   subset='train', # 加载那一部分数据集 train/test
                   categories=None, # 选取哪一类数据集[类别列表]，默认20类
                   #categories=['alt.atheism','comp.graphics','misc.forsale','rec.autos','sci.crypt','soc.religion.christian','talk.politics.guns'],
                   shuffle=True,  # 将数据集随机排序
                   random_state=42, # 随机数生成器
                   remove=(), # ('headers','footers','quotes') 去除标题、结尾或引用
                   download_if_missing=True # 如果没有下载过，重新下载
                   )
    #test=fetch_20newsgroups(categories=['alt.atheism','comp.graphics','misc.forsale','rec.autos','sci.crypt','soc.religion.christian','talk.politics.guns'],subset='test')
    test=fetch_20newsgroups(subset='test')
    return train,test

def TDIDF(train, test):
    #计算TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    train_v=vectorizer.fit_transform(train.data)
    test_v=vectorizer.transform(test.data)
    return train_v,test_v

'''
def pri(test,pred):
    print(classification_report(test.target, pred))
    print('f1_score:',f1_score(test.target,pred,average='macro'))
    print('accuracy_score:',accuracy_score(test.target,pred))
'''

def beiyesi(train,train_v,test_v):
    clf=[]
    pred=[]
    alpha=1.0
    for i in tqdm(range(5),colour='#00B2FF'):
        clf.append(MultinomialNB(alpha=alpha))
        clf[i].fit(train_v,train.target)
        pred.append(clf[i].predict(test_v))
        print(' alpha=',alpha,'f1_score:',f1_score(test.target,pred[i],average='macro'),'accuracy_score:',accuracy_score(test.target,pred[i]))
        alpha=alpha/10
    print(classification_report(test.target, pred[1]))
    print(classification_report(test.target, pred[2]))

def ganzhiji(train,train_v,test_v):
    clf=[]
    pred=[]
    alpha=['l2','l1','elasticnet','None']
    for i in tqdm(range(4),colour='#00B2FF'):
        if i < 3:
            clf.append(Perceptron(penalty=alpha[i]))
        else:
            clf.append(Perceptron())
        clf[i].fit(train_v,train.target)
        pred.append(clf[i].predict(test_v))
        print(' penalty=',alpha[i],'f1_score:',f1_score(test.target,pred[i],average='macro'),'accuracy_score:',accuracy_score(test.target,pred[i]))
    print(classification_report(test.target, pred[3]))

def k(train,train_v,test_v):
    clf=[]
    pred=[]
    alpha=1

    for i in tqdm(range(5),colour='#00B2FF'):
        clf.append(KNeighborsClassifier(n_neighbors=alpha))
        clf[i].fit(train_v,train.target)
        pred.append(clf[i].predict(test_v))
        print(' n_neighbors=',alpha,'f1_score:',f1_score(test.target,pred[i],average='macro'),'accuracy_score:',accuracy_score(test.target,pred[i]))
        alpha=alpha+1
    '''
    clf.append(KNeighborsClassifier(n_neighbors=alpha))
    clf[0].fit(train_v,train.target)
    pred.append(clf[0].predict(test_v))
    print(classification_report(test.target, pred[0]))
'''

def zhichi(train,train_v,test_v):
    clf=[]
    pred=[]
    alpha=0.0
    for i in tqdm(range(6),colour='#00B2FF'):
        if alpha==0.0:
            clf.append(SVC(C=0.1))
        else:
            clf.append(SVC(C=alpha))
        clf[i].fit(train_v,train.target)
        pred.append(clf[i].predict(test_v))
        if alpha==0.0:
            print(' C=0.1','f1_score:',f1_score(test.target,pred[i],average='macro'),'accuracy_score:',accuracy_score(test.target,pred[i]))
        else:
            print(' C=%.1f'%alpha,'f1_score:',f1_score(test.target,pred[i],average='macro'),'accuracy_score:',accuracy_score(test.target,pred[i]))
        alpha=alpha+0.2
'''
    for t in range(6):
        print(classification_report(test.target, pred[t]))
'''
def duocengganzhi(train,train_v,test_v):
    clf=[]
    pred=[]
    alpha=[[50,50],[50,100],[100,100]]
    for i in tqdm(range(3),colour='#00B2FF'):
        clf.append(MLPClassifier(hidden_layer_sizes=alpha[i]))
        clf[i].fit(train_v,train.target)
        pred.append((clf[i].predict(test_v)))
        print('f1_score:',f1_score(test.target,pred[i],average='macro'),'accuracy_score:',accuracy_score(test.target,pred[i]))
    for t in range(3):
        print(classification_report(test.target, pred[t]))


if __name__ == '__main__':
    train,test=predata()
    train_v,test_v=TDIDF(train, test)
    while True:
        x=int(input("请选择模型：1：贝叶斯，2：感知机，3：k近邻，4：支持向量机，5，多层感知机，0：退出\n"))
        if x==1:
            beiyesi(train,train_v,test_v)
        elif x==2:
            ganzhiji(train,train_v,test_v)
        elif x==3:
            k(train,train_v,test_v)
        elif x==4:
            zhichi(train,train_v,test_v)
        elif x==5:
            duocengganzhi(train,train_v,test_v)
        elif x==0:
            break
        else:
            print('输入错误！')

