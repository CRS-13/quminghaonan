import numpy as np

# 输入文件的路径
# x_train_path = '/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/train_bone.npy'
# y_train_path = '/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/train_label.npy'
# x_val_path = '/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_A_bone.npy'
# y_val_path = '/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_A_label.npy'
x_train_path = '/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/train_joint.npy'
y_train_path = '/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/train_label.npy'
x_val_path = '/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_B_joint.npy'
y_val_path = '/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/zero_label_B.npy'

# 加载数据
x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_val = np.load(x_val_path)
y_val = np.load(y_val_path)

# 保存为 .npz 文件
output_path = '/home/zjl_laoshi/xiaoke/UAV-SAR/data/uav/MMVRAC_CSv2_test_joint.npz'
np.savez(output_path, x_train=x_train, y_train=y_train, x_test=x_val, y_test=y_val)

print(f'数据已保存到 {output_path}')
