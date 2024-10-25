# 取名好难
这是我们的比赛代码，所有的训练权重全部保存在百度网盘
[baidupan]([https://pan.baidu.com/s/1kourPFzEChrjc8kPO0y6rw](https://pan.baidu.com/s/1ZG5e7c7tLNYy4-57Tl4dPg?pwd=szuq)),passwd is `szuq`

# TEGCN

## Data preparation
Prepare the data according to [https://github.com/CRS-13/quminghaonan/tree/4b5ec6abda2bc838f77200ff7f5e8a058286c46c/dataset/dataset].

Your `data/` should be like this:
```
dataset
└─data
    ├── train_label.npy
    ├── train_bone_motion.npy
    ├── train_bone.npy
    ├── train_joint_bone.npy
    ├── train_joint_motion.npy
    ├── train_joint.npy
    ├── test_*_bone_motion.npy
    ├── test_*_bone.npy
    ├── test_*_joint_bone.npy
    ├── test_*_joint_motion.npy
    ├── test_*_joint.npy
    ├── ..........
    ├── zero_label_B.npy
└─eval

```

## TRAIN
You can train the your model using the scripts:
```
sh scripts/TRAIN_V2.sh
```
注：应该检查训练的数据路径，在config文件中，我们使用该方法训练了四个模型分别使用joint、bone、joint_motion和bone_motion的数据。

## TEST
You can test the your model using the scripts:
```
sh scripts/EVAL_V2.sh
```
注：进行测试的时候需要修改测试结果保存路径，分别保存四个不同模型的测试结果。

## WEIGHTS
We have released all trained weights in [baidupan]([https://pan.baidu.com/s/1kourPFzEChrjc8kPO0y6rw](https://pan.baidu.com/s/1ZG5e7c7tLNYy4-57Tl4dPg?pwd=szuq)),passwd is `szuq`

# Top
它包含MixFormer和MixGCN

# Mixformer









