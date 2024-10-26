# 取名好难
这是我们的比赛代码，所有的训练权重全部保存在百度网盘,同时也在该项目的results目录下，同时还包含我们得到3dpose数据所用的附加文件
[baidupan]([ https://pan.baidu.com/s/1ZG5e7c7tLNYy4-57Tl4dPg?pwd=szuq]), passwd is `szuq`

# Install
执行下面命令：
```
cd Top
conda env create -f GCN.yml
conda env create -f 3dpose.yml
```
在转换3dpose时需要使用3dpose的环境，
TEGCN和Top的训练均使用GCN环境，在训练模型时如果缺少包，直接pip install 即可


# TEGCN

## Data preparation
Prepare the data according to [https://github.com/CRS-13/quminghaonan/tree/4b5ec6abda2bc838f77200ff7f5e8a058286c46c/dataset/dataset].

Your `dataset/` should be like this:
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
TE-GCN
Top
.....

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
We have released all trained weights in [baidupan]([ https://pan.baidu.com/s/1ZG5e7c7tLNYy4-57Tl4dPg?pwd=szuq]), passwd is `szuq`

# Top
它包含MixFormer和MixGCN

## Dataset
**1. 进入Top/Process_data，修改npy_to_npz.py代码中的路径,使处理后的数据保存在Top/Test_dataset/save_2d_pose
```
python npy_to_npz.py
```

**2. 得到3dpose数据
First, you must download the 3d pose checkpoint from [here](https://drive.google.com/file/d/1citX7YlwaM3VYBYOzidXSLHb4lJ6VlXL/view?usp=sharing), and install the environment based on **pose3d.yml** <br />
Then, you must put the downloaded checkpoint into the **./Process_data/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite** folder. <br />
你也可以在我们的网盘下载该文件，并放在上述指定的文件夹下
最后，你需要改变test_dataset_path和保存路径，得到joint和bone的3dpose数据
```
cd Process_data
python estimate.py --test_dataset_path ../Test_dataset
```
将得到的npz文件放入Top/Test_dataset/save_3d_pose下

# Model training
注：注意修改配置文件的数据路径，训练joint时使用joint的npz文件，在save_2d_pose文件夹下，3dpose在save_3d_pose下,test参数的data_path训练时使用A，测试时使用B
```
# Change the configuration file (.yaml) of the corresponding modality.
# Mix_GCN Example
cd ./Model_inference/Mix_GCN
python main.py --config ./config/ctrgcn_V1_J.yaml --device 0

# Mix_Former Example
cd ./Model_inference/Mix_Former
python main.py --config ./config/mixformer_V1_J.yaml --device 0
```
注意：我们训练了skmixf__V1_J、skmixf__V1_B、skmixf__V1_JM、skmixf__V1_BM、skmixf__V1_k2、skmixf__V1_k2M、
ctrgcn_V1_J、ctrgcn_V1_B、ctrgcn_V1_J_3D、ctrgcn_V1_B_3D、tdgcn_V1_J、tdgcn_V1_B、mstgcn_V1_J和mstgcn_V1_B模型

# Model inference
## Run Mix_GCN

```
cd ./Model_inference/Mix_GCN
pip install -e torchlight
```

**1. Run the following code separately to obtain classification scores using different model weights.** <br />
**test:**
注：在测试之前，需要将test的data_path改为的npz文件，注意joint与joint对应，bone与bone对应
在测试3dpose数据时，需要取消main.py文件中的446行代码， label = label.unsqueeze(1)
```
python main.py --config ./config/ctrgcn_V1_J.yaml --phase test --save-score True --weights ./your_pt_path/ctrgcn_V1_J.pt --device 0
python main.py --config ./config/ctrgcn_V1_B.yaml --phase test --save-score True --weights ./your_pt_path/ctrgcn_V1_B.pt --device 0
python main.py --config ./config/ctrgcn_V1_J_3d.yaml --phase test --save-score True --weights ./your_pt_path/ctrgcn_V1_J_3d.pt --device 0
python main.py --config ./config/ctrgcn_V1_B_3d.yaml --phase test --save-score True --weights ./your_pt_path/ctrgcn_V1_B_3d.pt --device 0
###
python main.py --config ./config/tdgcn_V1_J.yaml --phase test --save-score True --weights ./your_pt_path/tdgcn_V1_J.pt --device 0
python main.py --config ./config/tdgcn_V1_B.yaml --phase test --save-score True --weights ./your_pt_path/tdgcn_V1_B.pt --device 0
###
python main.py --config ./config/mstgcn_V1_J.yaml --phase test --save-score True --weights ./your_pt_path/mstgcn_V1_J.pt --device 0
python main.py --config ./config/mstgcn_V1_B.yaml --phase test --save-score True --weights ./your_pt_path/mstgcn_V1_B.pt --device 0
```

## Run Mix_Former

```
cd ./Model_inference/Mix_Former
```
**1. Run the following code separately to obtain classification scores using different model weights.** <br />
**CSv1:** <br />
You have to change the corresponding **data-path** in the **config file**, just like：**data_path: dataset/save_2d_pose/V1.npz**. we recommend using an absolute path.
注：与MixGCN一样需要检查数据路径，test参数的data_path训练时使用A，测试时使用B
```
python main.py --config ./config/mixformer_V1_J.yaml --phase test --save-score True --weights ./your_pt_path/mixformer_V1_J.pt --device 0  
python main.py --config ./config/mixformer_V1_B.yaml --phase test --save-score True --weights ./your_pt_path/mixformer_V1_B.pt --device 0 
python main.py --config ./config/mixformer_V1_JM.yaml --phase test --save-score True --weights ./your_pt_path/mixformer_V1_JM.pt --device 0 
python main.py --config ./config/mixformer_V1_BM.yaml --phase test --save-score True --weights ./your_pt_path/mixformer_V1_BM.pt --device 0 
python main.py --config ./config/mixformer_V1_k2.yaml --phase test --save-score True --weights ./your_pt_path/mixformer_V1_k2.pt --device 0 
python main.py --config ./config/mixformer_V1_k2M.yaml --phase test --save-score True --weights ./your_pt_path/mixformer_V1_k2M.pt --device 0 
```

# Ensemble

## Ensemble Mix_GCN、Mix_Former和TEGCN

**1.** You can obtain the final classification accuracy of CSv1 by running the following code:
```
python Ensemble_B.py
注意：在运行上述指令前，需要该修改文件中的路径,并按默认值对应的结果依次更改new_test_r1_Score到new_test_r18_Score

当然，我们的网盘中提供了我们的结果，在融合我们的结果前也需要修改成对应的路径

# Contact
如果复现过程中存在问题，可以通过QQ或者邮件联系我们，谢谢！
QQ：3091166956
email:3091166956@qq.com

