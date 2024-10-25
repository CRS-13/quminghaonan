'''
@File    :   estimate_3dpose.py
@Time    :   2024/03/23 4:00:00
@Author  :   Jinfu Liu
@Version :   1.0 
@Desc    :   estimate 3d pose from 2d pose
'''

import json
import copy
import argparse
import numpy as np
from lib.utils.tools import *
from lib.utils.learning import *
import os

CS_train_V1 = [0, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                21, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 57, 59, 
                61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 73, 76, 77, 78, 79, 80, 
                81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 98, 100, 102, 103, 105, 
                106, 110, 111, 112, 114, 115, 116, 117, 118]

CS_train_V2 = [0, 3, 4, 5, 6, 8, 10, 11, 12, 14, 16, 18, 19, 20, 21, 22, 24, 
                26, 29, 30, 31, 32, 35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 
                49, 52, 54, 56, 57, 59, 60, 61, 62, 63, 64, 66, 67, 69, 70, 71, 
                72, 73, 74, 75, 77, 78, 79, 80, 81, 83, 84, 86, 87, 88, 89, 91, 
                92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104, 106, 107, 
                108, 109, 110, 111, 112, 113, 114, 115, 117, 118]

def load_npy_data(data_path):
    if data_path.endswith('.npz'):
        data = np.load(data_path, allow_pickle=True)
        return data['x'], data['y']  # Assuming the .npz files have 'x' and 'y' keys.
    elif data_path.endswith('.npy'):
        data = np.load(data_path, allow_pickle=True)
        return data[:, 0], data[:, 1]  # Adjust this line based on your .npy structure.
    else:
        raise ValueError("Unsupported file format: {}".format(data_path))
    

def crop_scale(motion, scale_range=[1, 1]):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]!=0][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape)
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape)
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2] - [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    return result

def extract_pose(ske_txt_path: str) -> np.ndarray:
    with open(ske_txt_path, 'r') as f: 
        num_frame = int(f.readline()) # the frame num
        joint_data = [] # T M V C
        for t in range(num_frame): # for each frame
            num_body = int(f.readline()) # the body num
            one_frame_data = np.zeros((num_body, 17, 2)) # M 17 2 
            for m in range(num_body): # for each body
                f.readline() # skip this line, e.g. 000 0 0 0 0 0 0 0 0 0
                num_joints = int(f.readline()) # the num joins, equal to 17
                assert num_joints == 17
                for v in range(num_joints): # for each joint
                    xy = np.array(f.readline().split()[:2], dtype = np.float64)
                    one_frame_data[m, v] = xy
            joint_data.append(one_frame_data)
        joint_data = np.array(joint_data)  
    return joint_data # T M 17 2 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, default = "./configs/pose3d/MB_ft_h36m_global_lite.yaml", help = "Path to the config file.")
    parser.add_argument('-e', '--evaluate', default = './checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument(
        '--test_dataset_path', 
        type = str,
        default = '../Test_dataset'), # It's better to use absolute paths.
    opts = parser.parse_args()
    return opts

# python estimate_3dpose.py --test_dataset_path ../Test_dataset
if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    model_backbone = load_backbone(args)
    
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()
    
    print('Loading checkpoint', opts.evaluate)
    checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
    model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    model_pos = model_backbone
    model_pos.eval()

    root_train_path = os.path.join(opts.test_dataset_path, 'train')
    root_test_path = os.path.join(opts.test_dataset_path, 'test')
    
    train_data, train_labels = load_npy_data(os.path.join(root_train_path, 'train_joint.npy'))  # 更新为实际文件名
    test_data, test_labels = load_npy_data(os.path.join(root_test_path, 'test_A_joint.npy'))    # 更新为实际文件名

    # 保存结果的列表
    CS_train_data = []
    CS_train_label = []
    CS_test_data = []
    CS_test_label = []

    # 处理训练数据
    for idx, (joint_data, label) in enumerate(zip(train_data, train_labels)):
        print("Processing training sample", idx)
        # data = np.ones((joint_data.shape[0], joint_data.shape[1], joint_data.shape[2], 3))
        # data[:, :, :, :2] = joint_data
        data = np.zeros((joint_data.shape[1], joint_data.shape[0], 17, 3))  # 创建一个 (M, T, 17, 3) 的四维数组
        data[:, :, :, :2] = joint_data  # 将前两个维度赋值
        data = data.transpose(1, 0, 2, 3)
        data = crop_scale(data)

        pre_3d_pose = torch.zeros(2, 243, 17, 3)
        with torch.no_grad():
            if torch.cuda.is_available():
                data_input = torch.from_numpy(data).float().cuda()
            
            if data_input.shape[1] >= 243:
                data_input = data_input[:, :243, :, :]
            
            if data_input.shape[0] > 1:
                for idx in range(2):
                    predicted_3d_pos = model_pos(data_input[idx:idx + 1])
                    pre_3d_pose[idx:idx + 1, :predicted_3d_pos.shape[1], :, :] = predicted_3d_pos
            else:
                predicted_3d_pos = model_pos(data_input)
                pre_3d_pose[0:1, :predicted_3d_pos.shape[1], :, :] = predicted_3d_pos
        
        CS_train_data.append(pre_3d_pose)
        CS_train_label.append(label)

    # 处理测试数据
    for idx, (joint_data, label) in enumerate(zip(test_data, test_labels)):
        print("Processing test sample", idx)
        # data = np.ones((joint_data.shape[0], joint_data.shape[1], joint_data.shape[2], 3))
        # data[:, :, :, :2] = joint_data
        data = np.zeros((joint_data.shape[1], joint_data.shape[0], 17, 3))  # 创建一个 (M, T, 17, 3) 的四维数组
        data[:, :, :, :2] = joint_data  # 将前两个维度赋值
        data = data.transpose(1, 0, 2, 3)
        data = crop_scale(data)

        pre_3d_pose = torch.zeros(2, 243, 17, 3)
        with torch.no_grad():
            if torch.cuda.is_available():
                data_input = torch.from_numpy(data).float().cuda()
            
            if data_input.shape[1] >= 243:
                data_input = data_input[:, :243, :, :]
            
            if data_input.shape[0] > 1:
                for idx in range(2):
                    predicted_3d_pos = model_pos(data_input[idx:idx + 1])
                    pre_3d_pose[idx:idx + 1, :predicted_3d_pos.shape[1], :, :] = predicted_3d_pos
            else:
                predicted_3d_pos = model_pos(data_input)
                pre_3d_pose[0:1, :predicted_3d_pos.shape[1], :, :] = predicted_3d_pos
        
        CS_test_data.append(pre_3d_pose)
        CS_test_label.append(label)

    # 保存训练集和测试集结果
    # np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_3d_pose/train_results.npz', x_train=np.array(CS_train_data), y_train=np.array(CS_train_label))
    np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_3d_pose/test_B_bone_motion.npz', x_test=np.array(CS_test_data), y_test=np.array(CS_test_label))
    
    print("All done!")
    
    
    # opts = parse_args()
    # args = get_config(opts.config)
    # model_backbone = load_backbone(args)
    
    # if torch.cuda.is_available():
    #     model_backbone = nn.DataParallel(model_backbone)
    #     model_backbone = model_backbone.cuda()
    
    # print('Loading checkpoint', opts.evaluate)
    # checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
    # model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    # model_pos = model_backbone
    # model_pos.eval()

    # root_train_path = os.path.join(opts.test_dataset_path, 'train')
    # root_test_path = os.path.join(opts.test_dataset_path, 'test')
    
    # # Load data (adjust filenames as necessary)
    # train_data, train_labels = load_data(os.path.join(root_train_path, 'train_joint.npy'))
    # test_data, test_labels = load_data(os.path.join(root_test_path, 'test_A_joint.npy'))

    # # Preprocess and infer
    # for idx, (joint_data, label) in enumerate(zip(test_data, test_labels)):
    #     print("Processing sample", idx)
    #     # data = np.ones((joint_data.shape[0], joint_data.shape[1], joint_data.shape[2], 3))
    #     # data[:, :, :, :2] = joint_data
    #     data = np.zeros((joint_data.shape[1], joint_data.shape[0], 17, 3))  # 创建一个 (M, T, 17, 3) 的四维数组
    #     data[:, :, :, :2] = joint_data  # 将前两个维度赋值

    #     data = data.transpose(1, 0, 2, 3)
    #     data = crop_scale(data)

    #     pre_3d_pose = torch.zeros(2, 243, 17, 3)
    #     with torch.no_grad():
    #         if torch.cuda.is_available():
    #             data_input = torch.from_numpy(data).float().cuda()
            
    #         if data_input.shape[1] >= 243:
    #             data_input = data_input[:, :243, :, :]
            
    #         if data_input.shape[0] > 1:
    #             for idx in range(2):
    #                 predicted_3d_pos = model_pos(data_input[idx:idx + 1])
    #                 pre_3d_pose[idx:idx + 1, :predicted_3d_pos.shape[1], :, :] = predicted_3d_pos
    #         else:
    #             predicted_3d_pos = model_pos(data_input)
    #             pre_3d_pose[0:1, :predicted_3d_pos.shape[1], :, :] = predicted_3d_pos
        
    #     # Save or use the predicted 3D pose as needed

    # np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_3d_pose/test_results.npz', x_test=pre_3d_pose, y_test=test_labels)
    # print("All done!")
