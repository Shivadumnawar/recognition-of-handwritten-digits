# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:57:36 2020

@author: shiva dumnawar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
digits= load_digits()

dir(digits)

df= pd.DataFrame(digits.data, columns= digits.feature_names)

print(digits.images[0])

print(plt.matshow(digits.images[0]))

df['target']= digits.target

df.info()

X= df.iloc[:, :-1]
y= df['target'].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

from sklearn.svm import SVC
# svm with rbf kernel

clf= SVC()

clf.fit(X_train, y_train.ravel())

pred= clf.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 

print(confusion_matrix(y_test, pred))

print(accuracy_score(y_test, pred))

print(classification_report(y_test, pred))


# svm with linear kernel

model = SVC(kernel= 'linear')

model.fit(X_train, y_train.ravel())

y_pred= model.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

