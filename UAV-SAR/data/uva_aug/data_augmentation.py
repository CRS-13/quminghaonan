import numpy as np
import os

def time_shift(data, shift):
    shifted_data = np.roll(data, shift, axis=2)
    if shift > 0:
        shifted_data[:, :, :shift, :, :] = data[:, :, 0:1, :, :]
    elif shift < 0:
        shifted_data[:, :, shift:, :, :] = data[:, :, -1:, :, :]
    return shifted_data

def noise_injection(data, scale=0.01):
    noise = np.random.normal(loc=0.0, scale=scale, size=data.shape)
    return data + noise

def random_crop_and_padding(data, crop_size):
    N, C, T, V, M = data.shape
    start = np.random.randint(0, T - crop_size)
    cropped_data = data[:, :, start:start+crop_size, :, :]
    padded_data = np.pad(cropped_data, ((0, 0), (0, 0), (0, T - cropped_data.shape[2]), (0, 0), (0, 0)), mode='edge')
    return padded_data

def random_scaling(data, min_scale=0.5, max_scale=1.5):
    scale = np.random.uniform(min_scale, max_scale)
    return data * scale

def random_slicing(data, slice_length):
    N, C, T, V, M = data.shape
    start = np.random.randint(0, T - slice_length)
    return data[:, :, start:start+slice_length, :, :]

def augment_data(data, label):
    augmented_samples = []
    augmented_labels = []
    
    for i in range(data.shape[0]):
        sample = data[i:i+1]
        augmented_samples.append(sample)
        augmented_labels.append(label[i:i+1])  # 记录原始标签
        original_length = sample.shape[2]

        for _ in range(2):
            augmented_sample = sample.copy()
            augmented_label = label[i:i+1]  # 保持标签不变

            if np.random.rand() > 0.5:
                augmented_sample = time_shift(augmented_sample, shift=np.random.randint(-3, 4))
            if np.random.rand() > 0.5:
                augmented_sample = noise_injection(augmented_sample)
            if np.random.rand() > 0.5:
                crop_size = np.random.randint(200, original_length)
                augmented_sample = random_crop_and_padding(augmented_sample, crop_size=crop_size)
            if np.random.rand() > 0.5:
                augmented_sample = random_scaling(augmented_sample)
            if np.random.rand() > 0.5:
                slice_length = np.random.randint(50, 100)
                augmented_sample = random_slicing(augmented_sample, slice_length=slice_length)

            if augmented_sample.shape[2] != original_length:
                augmented_sample = np.pad(augmented_sample, ((0, 0), (0, 0), (0, original_length - augmented_sample.shape[2]), (0, 0), (0, 0)), mode='edge')

            augmented_samples.append(augmented_sample)
            augmented_labels.append(augmented_label)  # 记录对应的标签

    return np.concatenate(augmented_samples, axis=0), np.concatenate(augmented_labels, axis=0)

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.npz'):
            filepath = os.path.join(input_folder, filename)
            data = np.load(filepath)
            x_train = data['x_train']
            y_train = data['y_train']
            x_test = data['x_test']
            y_test = data['y_test']
            
            augmented_train, augmented_labels = augment_data(x_train, y_train)
            # augmented_test, augmented_test_labels = augment_data(x_test, y_test)
            
            output_filepath = os.path.join(output_folder, filename)
            np.savez(output_filepath, x_train=augmented_train, y_train=augmented_labels, x_test=x_test, y_test=y_test)

# 使用示例
input_folder = '/home/zjl_laoshi/xiaoke/UAV-SAR/data'  # 输入文件夹路径
output_folder = '/home/zjl_laoshi/xiaoke/UAV-SAR/data/uva_aug'  # 输出文件夹路径
process_folder(input_folder, output_folder)
