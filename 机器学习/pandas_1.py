import pandas as pd
from IPython.display import display

# 创建关于人的简单数据集
data = {'Name':["John","Anna","Peter","Linda"],
        'Location':["New York","Paris","Berlin","London"],
        'Age':[24,13,53,33]
        }

data_panda = pd.DataFrame(data)
display(data_panda)

display(data_panda[data_panda.Age>30])