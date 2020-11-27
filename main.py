import numpy as np
import math as mt
from io import StringIO
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import warnings
warnings.filterwarnings("ignore")

def importData():
    file = open("wine.data", "r")
    fileContent = StringIO(file.read())
    dataArr = np.genfromtxt(fileContent, delimiter=",")
    indexesArr = dataArr.copy()
    y = np.delete(dataArr, slice(1, 14), 1)
    y = y.astype(np.int32)
    X = np.delete(dataArr, 0, 1)
    arr=[X,y]
    return arr
def discretize(n,X):
    est = KBinsDiscretizer(n_bins=n, encode='ordinal', strategy='uniform')
    est.fit(X)
    discData = est.transform(X)
    return discData

def accuracy(predictresult, ytest):
    lz=0
    for i in range(len(ytest)):
        if ytest[i]==predictresult[i]:
            lz+=1

    return (lz/(len(ytest)))*100

class naiveBayesClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,laplace):
        self.laplace=laplace
        self.apriori=[]
        self.probabilities=[]

    def fit(self, X, y):
        counts = np.unique(y, return_counts=True)
        countsX  = np.unique(X, return_counts=True)[0]
        countsX = countsX.astype(np.int32)
        arr=counts[0]
        arr=arr.astype(np.int32)
        frequency=counts[1]
        for i in range(len(frequency)):
            self.apriori.append(frequency[i]/len(y))
        for m in arr:
            for l in range(X.shape[1]):
                for k in countsX:
                    counter=0
                    for n in range(len(X)):
                        if (int)(y[n])==m and (int)(X[n,l])==k:
                            counter+=1
                    if self.laplace==0:
                        condProb=counter/frequency[m-1]
                        self.probabilities.append(condProb)
                    else:
                        condProb = (counter+1) / (frequency[m - 1]+len(countsX))
                        self.probabilities.append(condProb)
    def predict(self,X):
        countsX = np.unique(X, return_counts=True)[0]
        countsX = countsX.astype(np.int32)
        result=[]
        for i in range(len(X)):
            helpList=[]
            for j in range(len(self.apriori)):
                counter=1
                for k in range(X.shape[1]):
                    index=j*13*3+k*3+(int)(X[i][k])
                    counter=counter*self.probabilities[index]
                counter=counter*self.apriori[j]
                helpList.append(counter)
            result.append(np.argmax(helpList)+1)
        return result

    def predictProba(self,X):
        countsX = np.unique(X, return_counts=True)[0]
        countsX = countsX.astype(np.int32)
        result = []
        for i in range(len(X)):
            helpList = []
            for j in range(len(self.apriori)):
                counter = 1
                for k in range(X.shape[1]):
                    index = j * 13 * 3 + k * 3 + (int)(X[i][k])
                    counter = counter * self.probabilities[index]
                counter = counter * self.apriori[j]
                helpList.append(counter)
            sumList=np.sum(helpList)
            for p in range(len(helpList)):
                helpList[p]=helpList[p]/sumList
            result.append(helpList)
        return result

class naiveBayesClassifierContinuous(BaseEstimator,ClassifierMixin):

    def __init__(self):
        self.averages=[]
        self.standardDeviation=[]

    def fit(self, X,y):
        counts = np.unique(y, return_counts=True)
        arr = counts[0]
        arr = arr.astype(np.int32)
        frequency = counts[1]
        for i in arr:
            for j in range(X.shape[1]):
                counter = 0
                avg = 0
                for k in range(len(X)):
                    if y[k] == i:
                        counter=counter+X[k][j]
                avg=counter/frequency[i-1]
                self.averages.append(avg)
        index=0
        for i in arr:
            for j in range(X.shape[1]):
                counter = 0
                for k in range(len(X)):
                    if y[k]==i:
                        counter=counter+(X[k][j]-self.averages[index])**2
                        index+=1
                    sD=mt.sqrt(counter/frequency[i-1])
                    self.standardDeviation.append(sD)

if __name__ == '__main__':
    data=importData()
    X=data[0]
    y=data[1]
    X=discretize(3,X)
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    #bayes=naiveBayesClassifier(0)
    #bayes.fit(X_train,y_train)
    #predictResult=bayes.predict(X_test)
    #predictProbaResult=bayes.predictProba(X_test)
    #acc=accuracy(y_test, predictResult)
    #print(acc)
    #bayes1 = naiveBayesClassifier(1)
    #bayes1.fit(X_train, y_train)
    #predictResult = bayes1.predict(X_test)
    #predictProbaResult = bayes1.predictProba(X_test)
    #acc = accuracy(y_test, predictResult)
    #print(acc)
    bayesContinuous=naiveBayesClassifierContinuous()
    bayesContinuous.fit(X_train, y_train)




