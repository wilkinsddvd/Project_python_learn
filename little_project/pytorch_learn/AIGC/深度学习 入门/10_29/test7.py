from sklearn.model_selection import train_test_split
import numpy as np
import mglearn

X,y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train.shape: {}".format(X_train.shape))