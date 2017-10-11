from numpy import genfromtxt
import sys
import numpy as np
from random import randint
import PIL.Image
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def get_input_data(path):

	X_train = genfromtxt(path + 'notMNIST_train_data.csv', delimiter=',')
	y_train = genfromtxt(path + 'notMNIST_train_labels.csv', delimiter=',')
	X_test = genfromtxt(path + 'notMNIST_test_data.csv', delimiter=',')
	y_test = genfromtxt(path + 'notMNIST_test_labels.csv', delimiter=',')

	X_train = X_train.astype(float)
	y_train = y_train.astype(float)
	X_test = X_test.astype(float)
	y_test = y_test.astype(float)
	return X_train, y_train, X_test, y_test


path = sys.argv[1]
X_train, y_train, X_test, y_test = get_input_data(path)

scaler = StandardScaler()
scaler.fit(X_train)
scaler.fit(X_test)

clf = LogisticRegression(penalty='l2', C=0.001)
clf.fit(X_train, y_train)

test_preds = clf.predict(X_test)

for i in xrange(len(test_preds)):
	if test_preds[i] >= 0.5:
		test_preds[i] = 1
	else:
		test_preds[i] = 0

for i in xrange(len(test_preds)):
	print int(test_preds[i])

print accuracy_score(y_test, test_preds)

weights = clf.coef_
bias = clf.intercept_

weights = ((weights - np.mean(weights)) / (np.max(weights) - np.min(weights)))

weights = (weights + 1)/2

im = weights.reshape((28, 28))

plt.imshow(im)
plt.show()
