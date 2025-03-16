import numpy as np
from scipy import sparse

# 创建一个二维numpy数组，对角线wield1，其余都为0
eye = np.eye(4)
print("Numpy array:\n{}".format(eye))

# 将numpy数组转换为CSR格式的scipy稀疏矩阵

sparse_matrix = sparse.csr_matrix(eye)
print("\nscipy sparse CSR matrix:\n{}".format(sparse_matrix))

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)

eye_coo = sparse.coo_matrix((data,(row_indices,col_indices)))
print("COO representation:\n{}".format(eye_coo))
