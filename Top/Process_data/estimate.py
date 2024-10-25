import json
import copy
import argparse
import numpy as np
from lib.utils.tools import *
from lib.utils.learning import *

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
        default = ''), # It's better to use absolute paths. like this '/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_2d_pose/joint.npz'
    opts = parser.parse_args()
    return opts

# python estimate_3dpose.py --test_dataset_path ../Test_dataset
if __name__ == "__main__":
    # load model
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
    
    # 加载 npz 数据集
    npz_data = np.load(opts.test_dataset_path)
    x_data = npz_data['x_train']  # 这里根据你的 npz 文件结构获取数据
    y_data = npz_data['y_train']  # 这里根据你的 npz 文件结构获取标签
    
    CS_test_data = []
    CS_test_label = []
    
    for idx in range(x_data.shape[0]):
        print("process sample ", idx)
        
        joint_data = x_data[idx]  # 获取每个样本的 2D joint 数据
        label = y_data[idx]  # 获取相应的标签

        # 数据处理（同之前的方式）
        data = np.ones((joint_data.shape[0], joint_data.shape[1], joint_data.shape[2], 3))  # T M 17 3
        data[:, :, :, :2] = joint_data  # T M 17 3
        data = data.transpose(1, 0, 2, 3)  # M T 17 3
        data = crop_scale(data)

        # infer
        pre_3d_pose = torch.zeros(2, 243, 17, 3)  # M max_T(motion bert) V C
        with torch.no_grad():
            if torch.cuda.is_available():
                data_input = torch.from_numpy(data).float().cuda()  # M T V C

            if data_input.shape[1] >= 243:
                data_input = data_input[:, :243, :, :]  # clip_len 243

            if data_input.shape[0] > 1:  # Two body
                for idx in range(2):
                    predicted_3d_pos = model_pos(data_input[idx:idx + 1])
                    pre_3d_pose[idx:idx + 1, :predicted_3d_pos.shape[1], :, :] = predicted_3d_pos
            else:  # one body
                predicted_3d_pos = model_pos(data_input)
                pre_3d_pose[0:1, :predicted_3d_pos.shape[1], :, :] = predicted_3d_pos

        # 保存结果
        CS_test_data.append(pre_3d_pose)  # M T V C
        CS_test_label.append(label)

    # 转换为数组并保存
    CS_test_data = np.array(CS_test_data)
    CS_test_label = np.array(CS_test_label)

    np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_3d_pose/test_B_joint.npz', x_train=CS_test_data, y_train=CS_test_label)
    # np.savez('/home/zjl_laoshi/xiaoke/Top/Test_dataset/save_3d_pose/joint.npz', x_test=CS_test_data, y_test=CS_test_label) ##根据训练集或测试集选择

    print("All done!")