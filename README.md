# 取名好难
这是我们的比赛文件

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
sh scripts/TRAIN_V1.sh
sh scripts/TRAIN_V2.sh
```

## TEST
You can test the your model using the scripts:
```
sh scripts/EVAL_V1.sh
sh scripts/EVAL_V2.sh
```

## WEIGHTS
We have released two trained weights in [baidupan](https://pan.baidu.com/s/1kourPFzEChrjc8kPO0y6rw),passwd is `nwhu`

Your should put them into `runs/`.

- V1:TOP1-42.37%
- V2:TOP1-68.11%


