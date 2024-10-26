import numpy as np  
  
# 读取原始的 .npy 文件  
original_data = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_A_label.npy')  
  
# 确保原始数据的大小为 (2000, 155)  
assert original_data.shape == (2000, ), "Original data shape is not (2000, 155)"  
  
# 创建一个新的 NumPy 数组，大小为 (4599,)  
new_data = np.zeros((4599, ), dtype=original_data.dtype)  
  
# 将原始数据复制到新数组的前 2000 行  
# new_data[:2000, :] = original_data  
  
# 保存新数组为 .npy 文件  
np.save('zero_label_A.npy', new_data)  
  
print("New .npy file created successfully.")