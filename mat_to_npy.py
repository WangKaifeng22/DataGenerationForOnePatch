import scipy.io as io
import numpy as np
# 读取 .mat 文件
mat_data = io.loadmat(r'C:\Users\Administrator\Documents\Pre_Master_learn\graduationThesis\dataset\SoSMap\2.0e-03andInc2.0e-03\sample_000001.mat')

data_array = mat_data['SoSMap']
# 保存为 .npy 文件
np.save('sample_000001.npy', data_array)