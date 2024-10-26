import numpy as np

x = np.load('data/test_B_joint.npy',mmap_mode='r')
print(x.shape)

x = np.load('data/test_A_label.npy',mmap_mode='r')
print(x.shape)