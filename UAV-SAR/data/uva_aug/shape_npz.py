import numpy as np

def read_npz_and_print_shape(file_path):
    # 加载 npz 文件
    data = np.load(file_path, allow_pickle=True)

    # 获取 data 的形状
    if 'x_train' in data:
        data_array = data['x_train']
        label_array = data['y_test']
    elif 'x_test' in data:
        data_array = data['x_test']
    else:
        raise ValueError("The npz file does not contain 'x_train' or 'x_test'.")

    print(f"Shape of data: {data_array.shape}")
    print(f"Shape of label: {label_array.shape}")

# 使用示例
file_path = '/home/zjl_laoshi/xiaoke/UAV-SAR/data/uav/MMVRAC_CSv2_test.npz'  # 替换为您的文件路径
read_npz_and_print_shape(file_path)
