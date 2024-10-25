import numpy as np

######################## train ###############  #将数据集的npy文件转换为需要的npz格式
# 加载joint.npy文件
data1 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/train_joint.npy')
label1 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/train_label.npy')  # 假设标签文件名为train_labels.npy

# 加载bone.npy文件
data2 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/train_bone.npy')
label2 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/train_label.npy')  # 假设标签文件名为train_labels.npy

# 加载joint_motion.npy文件
data3 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/train_joint_motion.npy')
label3 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/train_label.npy')  # 假设标签文件名为train_labels.npy

# 加载bone_motion.npy文件
data4 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/train_bone_motion.npy')
label4 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/train_label.npy')  # 假设标签文件名为train_labels.npy

# 保存为.npz文件
np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_2d_pose/joint.npz', x_train=data1, y_train=label1)
np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_2d_pose/bone.npz', x_train=data2, y_train=label2)
np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_2d_pose/joint_motion.npz', x_train=data3, y_train=label3)
np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_2d_pose/bone_motion.npz', x_train=data4, y_train=label4)


######################## test ############### 
# 加载joint.npy文件
data1 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_A_joint.npy')
print(data1.shape)
label1 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_A_label.npy')  # 假设标签文件名为train_labels.npy
print(label1.shape)
# 加载bone.npy文件
data2 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_A_bone.npy')
label2 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_A_label.npy')  # 假设标签文件名为train_labels.npy

# 加载joint_motion.npy文件
data3 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_A_joint_motion.npy')
label3 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_A_label.npy')  # 假设标签文件名为train_labels.npy

# 加载bone_motion.npy文件
data4 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_A_bone_motion.npy')
label4 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_A_label.npy')  # 假设标签文件名为train_labels.npy

# 保存为.npz文件
np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_2d_pose/test_A_joint.npz', x_train=data1, y_train=label1)
np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_2d_pose/test_A_bone.npz', x_train=data2, y_train=label2)
np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_2d_pose/test_A_joint_motion.npz', x_train=data3, y_train=label3)
np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_2d_pose/test_A_bone_motion.npz', x_train=data4, y_train=label4)

######################## test B ############### #用于得到B测试集上的置信度文件
# 加载joint.npy文件
data1 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_B_joint.npy')
print(data1.shape)
label1 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/zero_label_B.npy')  # 假设标签文件名为train_labels.npy
print(label1.shape)
# 加载bone.npy文件
data2 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_B_bone.npy')
label2 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/zero_label_B.npy')  # 假设标签文件名为train_labels.npy

# 加载joint_motion.npy文件
data3 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_B_joint_motion.npy')
label3 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/zero_label_B.npy')  # 假设标签文件名为train_labels.npy

# 加载bone_motion.npy文件
data4 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_B_bone_motion.npy')
label4 = np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/zero_label_B.npy')  # 假设标签文件名为train_labels.npy

# 保存为.npz文件
np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_2d_pose/test_B_joint.npz', x_train=data1, y_train=label1)
np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_2d_pose/test_B_bone.npz', x_train=data2, y_train=label2)
np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_2d_pose/test_B_joint_motion.npz', x_train=data3, y_train=label3)
np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_2d_pose/test_B_bone_motion.npz', x_train=data4, y_train=label4)