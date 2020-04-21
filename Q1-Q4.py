from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


X, Y = fetch_openml('mnist_784', version=1, data_home='./scikit_learn_data', return_X_y=True)
# make the value of pixels from [0, 255] to [0, 1] for further process
X = X / 255.
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)


# TODO:use logistic regression (Q1)
log_reg = LogisticRegression(max_iter=2000)
log_reg.fit(X_train, Y_train)
y_test = log_reg.predict(X_test)
y_train = log_reg.predict(X_train)
train_accuracy = metrics.accuracy_score(Y_train, y_train)
test_accuracy = metrics.accuracy_score(Y_test, y_test)
print('逻辑回归Training accuracy: %0.2f%%' % (train_accuracy*100))
print('逻辑回归Testing accuracy: %0.2f%%' % (test_accuracy*100))


# TODO:use naive bayes (Q2)
from sklearn.naive_bayes import BernoulliNB
naive_bayes_classifier = BernoulliNB()
naive_bayes_classifier.fit(X_train, Y_train)
y_train = naive_bayes_classifier.predict(X_train)
y_test = naive_bayes_classifier.predict(X_test)
train_accuracy = metrics.accuracy_score(Y_train, y_train)
test_accuracy = metrics.accuracy_score(Y_test, y_test)
print('朴素贝叶斯Training accuracy: %0.2f%%' % (train_accuracy*100))
print('朴素贝叶斯Testing accuracy: %0.2f%%' % (test_accuracy*100))


# TODO:use support vector machine (Q3)
from sklearn.svm import LinearSVC
SVM = LinearSVC()
SVM.fit(X_train, Y_train)
y_train = SVM.predict(X_train)
y_test = SVM.predict(X_test)
train_accuracy = metrics.accuracy_score(Y_train, y_train)
test_accuracy = metrics.accuracy_score(Y_test, y_test)
print('SVM Training accuracy: %0.2f%%' % (train_accuracy*100))
print('SVM Testing accuracy: %0.2f%%' % (test_accuracy*100))


# adjust SVM parameters (Q4)
SVM_2 = LinearSVC(C=0.75, max_iter=5000)
SVM_2.fit(X_train, Y_train)
y_train = SVM_2.predict(X_train)
y_test = SVM_2.predict(X_test)
train_accuracy = metrics.accuracy_score(Y_train, y_train)
test_accuracy = metrics.accuracy_score(Y_test, y_test)
print('Adjusted SVM Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Adjusted SVM Testing accuracy: %0.2f%%' % (test_accuracy*100))
