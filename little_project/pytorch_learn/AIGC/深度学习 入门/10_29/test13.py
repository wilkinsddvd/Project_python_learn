import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import mglearn
from sklearn.model_selection import train_test_split

fig,axes =plt.subplots(1,3,figsize=(15,4))

line = np.linspace(-3,3,1000).reshape(-1,1)
for n_neighbors,ax in zip([1,3,9],axes):

    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    X,y = mglearn.datasets.make_wave(n_samples=40)
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
    reg.fit(X_train,y_train)
    ax.plot(line,reg.predict(line))
    ax.plot(X_train,y_train,'^',c=mglearn.cm2(0),markersize=8)
    ax.plot(X_test,y_test,'v',c=mglearn.cm2(1),markersize=8)
    ax.set_title("{}neighbors(s)\n train score {:.2f} test score {:.2f}".format(n_neighbors,reg.score(X_train,y_train),reg.score(X_test,y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model","Training data","Test data"],loc="best")
plt.show()
