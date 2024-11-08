from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import mglearn

clf = KNeighborsClassifier(n_neighbors=3)
X,y =mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
