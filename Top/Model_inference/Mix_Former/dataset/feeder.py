# import torch
# import numpy as np
# import torch.nn.functional as F
# from torch.utils.data import Dataset

# from . import tools

# coco_pairs = [(1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7), (7, 1), (8, 6), (9, 7), (10, 8), (11, 9),
#                 (12, 6), (13, 7), (14, 12), (15, 13), (16, 14), (17, 15)]

# class Feeder(Dataset):
#     def __init__(self, data_path: str, data_split: str, p_interval: list=[0.95], window_size: int=64, bone: bool=False, vel: bool=False):
#         super(Feeder, self).__init__()
#         self.data_path = data_path
#         self.data_split = data_split
#         self.p_interval = p_interval
#         self.window_size = window_size
#         self.bone = bone
#         self.vel = vel
#         self.load_data()
        
#     # def load_data(self):
#     #     npz_data = np.load(self.data_path, allow_pickle=True)
#     #     print(npz_data.keys())
#     #     if self.data_split == 'train':
#     #         self.data = npz_data['x_train']
#     #         self.label = npz_data['y_train']
#     #         self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
#     #     else:
#     #         assert self.data_split == 'test'
#     #         self.data = npz_data['x_test']
#     #         self.label = npz_data['y_test']
#     #         self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
    
#     def load_data(self):
#         npz_data = np.load(self.data_path, allow_pickle=True)
#         print(npz_data.keys())
        
#         if self.data_split == 'train':
#             # 根据实际数据结构修改这里
#             self.data = npz_data['data']  # 假设 'data' 包含训练数据
#             self.label = npz_data['y_train']  # 确保'y_train'键存在于npz文件中
#             self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
#         else:
#             assert self.data_split == 'test'
#             self.data = npz_data['data']  # 假设 'data' 包含测试数据
#             self.label = npz_data['y_test']  # 确保'y_test'键存在于npz文件中
#             self.sample_name = ['test_' + str(i) for i in range(len(self.data))]

            
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
#         data_numpy = self.data[idx] # T M V C
#         label = self.label[idx]
#         data_numpy = torch.from_numpy(data_numpy).permute(3, 0, 2, 1) # C,T,V,M
#         data_numpy = np.array(data_numpy)
#         valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
#         if(valid_frame_num == 0): 
#             return np.zeros((2, 64, 17, 2)), label, idx
#         # reshape Tx(MVC) to CTVM
#         data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
#         if self.bone:
#             bone_data_numpy = np.zeros_like(data_numpy)
#             for v1, v2 in coco_pairs:
#                 bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
#             data_numpy = bone_data_numpy
#         if self.vel:
#             data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
#             data_numpy[:, -1] = 0

#         data_numpy = data_numpy - np.tile(data_numpy[:, :, 0:1, :], (1, 1, 17, 1)) # all_joint - 0_joint
#         return data_numpy, label, idx # C T V M
    
#     def top_k(self, score, top_k):
#         rank = score.argsort()
#         hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
#         return sum(hit_top_k) * 1.0 / len(hit_top_k)
    
# if __name__ == "__main__":
    # Debug
    # train_loader = torch.utils.data.DataLoader(
    #             dataset = Feeder(data_path = '/data-home/liujinfu/MotionBERT/pose_data/V1.npz', data_split = 'train'),
    #             batch_size = 4,
    #             shuffle = True,
    #             num_workers = 2,
    #             drop_last = False)
    
    # val_loader = torch.utils.data.DataLoader(
    #         dataset = Feeder(data_path = '/data-home/liujinfu/MotionBERT/pose_data/V1.npz', data_split = 'test'),
    #         batch_size = 4,
    #         shuffle = False,
    #         num_workers = 2,
    #         drop_last = False)
    
    # for batch_size, (data, label, idx) in enumerate(train_loader):
    #     data = data.float() # B C T V M
    #     label = label.long() # B 1
    #     print("pasue")
    
    
    
    
#################### New Code ###########################
import torch
import numpy as np
from torch.utils.data import Dataset
from . import tools

class Feeder(Dataset):
    def __init__(self, data_path: str, label_path: str, data_split: str, p_interval: list=[0.95], window_size: int=64, bone: bool=False, vel: bool=False, transform=None):
        super(Feeder, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.data_split = data_split
        self.p_interval = p_interval
        self.window_size = window_size
        self.bone = bone
        self.vel = vel
        self.load_data()
        self.transform=transform
        
    def load_data(self):
        npz_data = np.load(self.data_path, allow_pickle=True)
        # print("Data keys:", npz_data.keys())
        self.data = npz_data['data']
        # print("Data shape:", self.data.shape)  # 打印数据的形状

        # 加载标签
        label_data = np.load(self.label_path, allow_pickle=True)
        # print("Label keys:", label_data.keys())
        self.label = label_data['data']
        # print("Label shape:", self.label.shape)  # 打印标签的形状

        if self.label.ndim == 1:  # 如果标签是一维的
            self.label = self.label.reshape(-1, 1)  # 根据需要调整形状

        self.sample_name = ['sample_' + str(i) for i in range(len(self.data))]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        data_numpy = self.data[idx]  # T M V C
        data_tensor = torch.from_numpy(data_numpy).permute(0, 2, 3, 1)  # 转换并调整维度

        label = self.label[idx] if self.label is not None else torch.tensor(0, dtype=torch.long)  # 默认标签类型

        valid_frame_num = np.sum(data_numpy.sum(axis=0).sum(axis=-1).sum(axis=-1) != 0)

        if valid_frame_num == 0:
            return torch.zeros((3, 300, 17, 2)), label, idx  # 根据实际需要调整形状

        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)

        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in coco_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        data_numpy = data_numpy - np.tile(data_numpy[:, :, 0:1, :], (1, 1, 17, 1))  # all_joint - 0_joint

        data_tensor = torch.from_numpy(data_numpy).permute(0, 3, 1, 2)  # 转换为 [3, 300, 17, 2]

        return data_tensor, label, idx  # 返回数据、标签和索引


    def top_k(self, score, top_k):
        if self.label is None:
            return 0  # 没有标签时返回 0
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
