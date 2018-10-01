#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:20:33 2018

@author: sherry
"""


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plot

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/' 
                      'machine-learning-databases/wine/wine.data', header=None)
#read and split data
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
#Standardlize data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#EDA
print(df_wine.head())
print(df_wine.tail())
print(df_wine.describe())

for i in range(0,177):
    if df_wine.iat[i,0]==1:
        pcolor='red'
    elif df_wine.iat[i,0]==2:
        pcolor="blue"
    else:
        pcolor='green'
        
    dataRow=df_wine.iloc[i,0:13]
    dataRow.plot(color=pcolor)
    
plot.xlabel('Attribute Index')
plot.ylabel('Attribute Values')
plot.show()

#HeatMap
corMat=pd.DataFrame(df_wine.corr())
plot.pcolor(corMat)
plot.show()

#import seaborn as sns
#sns.pairplot(df_wine.iloc[:,range(14)], size=2.5)
#plot.tight_layout()
#plot.show()

print('\nUntransformed data with LR')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
slr = LinearRegression()
slr.fit(X_train_std, y_train)
#y_train_pred = slr.predict(X_train)
#y_test_pred = slr.predict(X_test)
train_scores =cross_val_score(estimator=slr,X=X_train_std,y=y_train)
test_scores =cross_val_score(estimator=slr,X=X_test_std,y=y_test)
print('CV accuracy scores for train data: %s' % train_scores)
print('CV accuracy for train data: %.3f +/- %.3f' % (np.mean(train_scores),np.std(train_scores)))
print('CV accuracy scores for test data: %s' % test_scores)
print('CV accuracy for test data: %.3f +/- %.3f' % (np.mean(test_scores),np.std(test_scores)))

print('\nUntransformed data with SVM')
from sklearn.svm import SVC
svm = SVC(kernel="linear")
svm.fit(X_train_std, y_train)
train_scores =cross_val_score(estimator=svm,X=X_train_std,y=y_train)
test_scores =cross_val_score(estimator=svm,X=X_test_std,y=y_test)
print('CV accuracy scores for train data: %s' % train_scores)
print('CV accuracy for train data: %.3f +/- %.3f' % (np.mean(train_scores),np.std(train_scores)))
print('CV accuracy scores for test data: %s' % test_scores)
print('CV accuracy for test data: %.3f +/- %.3f' % (np.mean(test_scores),np.std(test_scores)))

print('\nPCA data with LR')
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
slr.fit(X_train_pca, y_train)
train_scores =cross_val_score(estimator=slr,X=X_train_pca,y=y_train)
test_scores =cross_val_score(estimator=slr,X=X_test_pca,y=y_test)
print('CV accuracy scores for train data: %s' % train_scores)
print('CV accuracy for train data: %.3f +/- %.3f' % (np.mean(train_scores),np.std(train_scores)))
print('CV accuracy scores for test data: %s' % test_scores)
print('CV accuracy for test data: %.3f +/- %.3f' % (np.mean(test_scores),np.std(test_scores)))

print('\nPCA data with SVM')
from sklearn.svm import SVC
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
svm.fit(X_train_pca, y_train)
train_scores =cross_val_score(estimator=svm,X=X_train_pca,y=y_train)
test_scores =cross_val_score(estimator=svm,X=X_test_pca,y=y_test)
print('CV accuracy scores for train data: %s' % train_scores)
print('CV accuracy for train data: %.3f +/- %.3f' % (np.mean(train_scores),np.std(train_scores)))
print('CV accuracy scores for test data: %s' % test_scores)
print('CV accuracy for test data: %.3f +/- %.3f' % (np.mean(test_scores),np.std(test_scores)))

print('\nLDA data with LR')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
slr.fit(X_train_lda, y_train)
train_scores =cross_val_score(estimator=slr,X=X_train_lda,y=y_train)
test_scores =cross_val_score(estimator=slr,X=X_test_lda,y=y_test)
print('CV accuracy scores for train data: %s' % train_scores)
print('CV accuracy for train data: %.3f +/- %.3f' % (np.mean(train_scores),np.std(train_scores)))
print('CV accuracy scores for test data: %s' % test_scores)
print('CV accuracy for test data: %.3f +/- %.3f' % (np.mean(test_scores),np.std(test_scores)))

print('\nLDA data with SVM')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)
svm.fit(X_train_lda, y_train)
train_scores =cross_val_score(estimator=svm,X=X_train_lda,y=y_train)
test_scores =cross_val_score(estimator=svm,X=X_test_lda,y=y_test)
print('CV accuracy scores for train data: %s' % train_scores)
print('CV accuracy for train data: %.3f +/- %.3f' % (np.mean(train_scores),np.std(train_scores)))
print('CV accuracy scores for test data: %s' % test_scores)
print('CV accuracy for test data: %.3f +/- %.3f' % (np.mean(test_scores),np.std(test_scores)))

# Report the best parameters
from sklearn.model_selection import GridSearchCV
parameters = {'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1,1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X_train_std,y_train)
print("Best CV params", searcher.best_params_)

print('\nKPCA data with LR')
from sklearn.decomposition import KernelPCA
for i in [0.00001,0.0001,0.001,0.01,0.1,1]:
    print("           gamma=",i)
    scikit_kpca = KernelPCA(n_components=2,kernel='rbf',gamma=i)
    X_train_kpca = scikit_kpca.fit_transform(X_train_std)
    X_test_kpca=scikit_kpca.transform(X_test_std)
    slr.fit(X_train_kpca, y_train)
    train_scores =cross_val_score(estimator=slr,X=X_train_kpca,y=y_train)
    test_scores =cross_val_score(estimator=slr,X=X_test_kpca,y=y_test)
    print('CV accuracy scores for train data: %s' % train_scores)
    print('CV accuracy for train data: %.3f +/- %.3f' % (np.mean(train_scores),np.std(train_scores)))
    print('CV accuracy scores for test data: %s' % test_scores)
    print('CV accuracy for test data: %.3f +/- %.3f' % (np.mean(test_scores),np.std(test_scores)))

print('\nKPCA data with SVM')
for i in [0.0005,0.005,0.05,0.1,1]:
    print("           gamma=",i)
    scikit_kpca = KernelPCA(n_components=2,kernel='rbf',gamma=i)
    X_train_kpca = scikit_kpca.fit_transform(X_train_std)
    X_test_kpca=scikit_kpca.transform(X_test_std)
    svm.fit(X_train_kpca, y_train)
    train_scores =cross_val_score(estimator=svm,X=X_train_kpca,y=y_train)
    test_scores =cross_val_score(estimator=svm,X=X_test_kpca,y=y_test)
    print('CV accuracy scores for train data: %s' % train_scores)
    print('CV accuracy for train data: %.3f +/- %.3f' % (np.mean(train_scores),np.std(train_scores)))
    print('CV accuracy scores for test data: %s' % test_scores)
    print('CV accuracy for test data: %.3f +/- %.3f' % (np.mean(test_scores),np.std(test_scores)))



print('\nMy name is Sihan Li')
print('My NetId is sihanl2')
print('I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.')


