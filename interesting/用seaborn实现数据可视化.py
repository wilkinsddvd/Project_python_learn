import time
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

print("seaborn version {}".format(sns.__version__))
# R expand.grid() function in Python
# https://stackoverflow.com/a/12131385/1135316
def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}

ltt= ['little1']

methods=["m" + str(i) for i in range(1,21)]
#methods=['method 1', 'method 2', 'method 3', 'method 4']
#labels = ['label1','label2']
labels=["l" + str(i) for i in range(1,20)]

times = range(0,100,4)
data = pd.DataFrame(expandgrid(ltt,methods,labels, times, times))
data.columns = ['ltt','method','labels','dtsi','rtsi']
#data['nw_score'] = np.random.sample(data.shape[0])
data['nw_score'] = np.random.choice([0,1],data.shape[0])

labels_fill = {0:'red',1:'blue'}

del methods
del labels


cmap=ListedColormap(['red', 'blue'])

def facet(data, ax):
    data = data.pivot(index="dtsi", columns='rtsi', values='nw_score')
    ax.imshow(data, cmap=cmap)

def myfacetgrid(data, row, col, figure=None):
    rows = np.unique(data[row].values)
    cols = np.unique(data[col].values)

    fig, axs = plt.subplots(len(rows), len(cols),
                            figsize=(2*len(cols)+1, 2*len(rows)+1))


    for i, r in enumerate(rows):
        row_data = data[data[row] == r]
        for j, c in enumerate(cols):
            this_data = row_data[row_data[col] == c]
            facet(this_data, axs[i,j])
    return fig, axs


for lt in data.ltt.unique():

    with sns.plotting_context(font_scale=5.5):
        t = time.time()
        fig, axs = myfacetgrid(data[data.ltt==lt], row="labels", col="method")
        print(time.time()-t)
        for ax,method in zip(axs[0,:],data.method.unique()):
            ax.set_title(method, fontweight='bold', fontsize=12)
        for ax,label in zip(axs[:,0],data.labels.unique()):
            ax.set_ylabel(label, fontweight='bold', fontsize=12, rotation=0, ha='right', va='center')
        print(time.time()-t)
        fig.suptitle(lt, fontweight='bold', fontsize=12)
        fig.tight_layout()
        fig.subplots_adjust(top=0.8) # make some room for the title
        print(time.time()-t)
        fig.savefig(lt+'.png', dpi=300)
        print(time.time()-t)