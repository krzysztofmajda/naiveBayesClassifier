import numpy as np
from io import StringIO
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

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

def accuracy(np,lz):
    return (np/lz)*100


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
                        if (int)(y[l])==m and (int)(X[n,l])==k:
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
        for i in range(len(X)):
            for j in range(len(self.apriori)):
                counter=1
                for k in range(X.shape[1]):
                    index=j*13*3+k*3+(int)(X[i][k])
                    counter=counter*self.probabilities[index]
                counter=counter*self.apriori[j]

    def predictproba(selfsel,X):
        print("")

if __name__ == '__main__':
    data=importData()
    X=data[0]
    y=data[1]
    X=discretize(3,X)
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    bayes=naiveBayesClassifier(1)
    bayes.fit(X_train,y_train)
    bayes.predict(X_test)


