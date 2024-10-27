# 取名好难
这是我们的比赛代码，所有的训练权重全部保存在百度网盘,同时也在该项目的results目录下，同时还包含我们得到3dpose数据所用的附加文件
[baidupan]([  https://pan.baidu.com/s/1Jc_pwz2FVt_2SrN1wiQOTg?pwd=ai3e]), passwd is `ai3e`

我们使用的都为绝对路径，所以在训练前需要检查路径

# Install
执行下面命令：
```shell
cd Top
conda env create -f GCN.yml
conda env create -f 3dpose.yml
```
在转换3dpose时需要使用3dpose的环境，
TEGCN和Top的训练均使用GCN环境，在训练模型时如果缺少包，直接pip install 即可
使用GCN运行代码时可能出现` File "/home/zjl_laoshi/anaconda3/envs/mixgcn_test/lib/python3.10/site-packages/torchpack/config.py", line 4, in <module>`
将`from collections import Iterable`修改为`from collections.abc import Iterable`即可


# TEGCN

## Data preparation
Prepare the data according to [https://github.com/CRS-13/quminghaonan/blob/577ce2e663f4cb7ff56bdc53603329d54edff5ea/dataset/README.md].
下面提供了
# 2024-全球校园人工智能算法精英大赛-算法挑战赛-基于无人机的人体行为识别

### 依赖库

python ：`numpy tqdm`

注意：

1. 完整流程可以直接运行子文件夹下的`ipynb`
2. 国内注意PIP换源，命令为：`pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple`

## 数据准备（UAV-human骨架数据预处理）

### 注意
1. 在验证中，我们发现Windows多线程处理存在一些问题（包括内存占用异常增加），或者系统盘与工作区在一个存储介质上，也会存在系统IO耗尽导致死机卡顿。因此默认启用单线程
2. 为加快处理速度，也可以启用多线程，通常可带来2-4倍的性能提升。

**多线程处理**：在以下提到的命令中，添加`--use_mp True`即可。

### 流程
1. 将省赛数据集解压放入data文件夹下，修改名称为test_B_joint.npy
2. 数据集处理出bone模态数据：运行`python gen_modal.py --modal bone`得到bone模态数据
3. 数据集处理出motion模态数据：运行`python gen_modal.py --modal motion`得到motion模态的数据
4. 运行zero.py文件得到（4599，）的全零标签
5. 最终你会得到如下所展示的目录结构与文件

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
```shell
sh scripts/TRAIN_V2.sh
```
注：应该检查训练的数据路径，在config文件中，我们使用该方法训练了四个模型分别使用joint、bone、joint_motion和bone_motion的数据。

## TEST
You can test the your model using the scripts:
```shell
sh scripts/EVAL_V2.sh
```
注：进行测试的时候需要修改测试结果保存路径，分别保存四个不同模型的测试结果。
注意修改权重地址

## WEIGHTS
We have released all trained weights in [baidupan]([  https://pan.baidu.com/s/1Jc_pwz2FVt_2SrN1wiQOTg?pwd=ai3e]), passwd is `szuq`

# Top
它包含MixFormer和MixGCN

## Dataset
**1. 进入Top/Process_data，修改npy_to_npz.py代码中的路径,使处理后的数据保存在Top/Test_dataset/save_2d_pose，该文件夹需要自己创建，同时创建Top/Test_dataset/save_3d_pose
```shell
python npy_to_npz.py
```

**2. 得到3dpose数据
First, you must download the 3d pose checkpoint from [here](https://drive.google.com/file/d/1citX7YlwaM3VYBYOzidXSLHb4lJ6VlXL/view?usp=sharing), and install the environment based on **3dpose.yml** <br />
Then, you must put the downloaded checkpoint into the **./Process_data/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite** folder. <br />
你也可以在我们的网盘下载该文件，并放在上述指定的文件夹下
最后，你需要改变test_dataset_path和保存路径，得到joint和bone的3dpose数据
```shell
cd Process_data
python estimate.py --test_dataset_path ../Test_dataset
```
将得到的npz文件放入Top/Test_dataset/save_3d_pose下

# Model training
注：注意修改配置文件的数据路径，训练joint时使用joint的npz文件，在save_2d_pose文件夹下，3dpose在save_3d_pose下,test参数的data_path训练时使用A，测试时使用B

安装包
```shell
cd ./Model_inference/Mix_GCN
pip install -e torchlight
```

```shell
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

**1. Run the following code separately to obtain classification scores using different model weights.** <br />
**test:**
注：在测试之前，需要将test的data_path改为的npz文件，注意joint与joint对应，bone与bone对应
在测试3dpose数据时，需要取消注释main.py文件中的446行代码， `label = label.unsqueeze(1)`
```shell
python main.py --config ./config/ctrgcn_V1_J.yaml --phase test --save-score True --weights ./your_pt_path/pt_name.pt --device 0
python main.py --config ./config/ctrgcn_V1_B.yaml --phase test --save-score True --weights ./your_pt_path/pt_name.pt --device 0
python main.py --config ./config/ctrgcn_V1_J_3d.yaml --phase test --save-score True --weights ./your_pt_path/pt_name.pt --device 0
python main.py --config ./config/ctrgcn_V1_B_3d.yaml --phase test --save-score True --weights ./your_pt_path/pt_name.pt --device 0
###
python main.py --config ./config/tdgcn_V1_J.yaml --phase test --save-score True --weights ./your_pt_path/pt_name.pt --device 0
python main.py --config ./config/tdgcn_V1_B.yaml --phase test --save-score True --weights ./your_pt_path/pt_name.pt --device 0
###
python main.py --config ./config/mstgcn_V1_J.yaml --phase test --save-score True --weights ./your_pt_path/pt_name.pt --device 0
python main.py --config ./config/mstgcn_V1_B.yaml --phase test --save-score True --weights ./your_pt_path/pt_name.pt --device 0
```

## Run Mix_Former

```shell
cd ./Model_inference/Mix_Former
```
**1. Run the following code separately to obtain classification scores using different model weights.** <br />
**CSv1:** <br />
You have to change the corresponding **data-path** in the **config file**, just like：**data_path: dataset/save_2d_pose/V1.npz**. we recommend using an absolute path.
注：与MixGCN一样需要检查数据路径，test参数的data_path训练时使用A，测试时使用B
```shell
python main.py --config ./config/mixformer_V1_J.yaml --phase test --save-score True --weights ./your_pt_path/pt_name.pt --device 0  
python main.py --config ./config/mixformer_V1_B.yaml --phase test --save-score True --weights ./your_pt_path/pt_name.pt --device 0 
python main.py --config ./config/mixformer_V1_JM.yaml --phase test --save-score True --weights ./your_pt_path/pt_name.pt --device 0 
python main.py --config ./config/mixformer_V1_BM.yaml --phase test --save-score True --weights ./your_pt_path/pt_name.pt --device 0 
python main.py --config ./config/mixformer_V1_k2.yaml --phase test --save-score True --weights ./your_pt_path/pt_name.pt --device 0 
python main.py --config ./config/mixformer_V1_k2M.yaml --phase test --save-score True --weights ./your_pt_path/pt_name.pt --device 0 
```
# Mix_Former_sar
cd UAV-SAR
## Dependencies
* python == 3.8
* pytorch == 1.1.3
* NVIDIA apex
* PyYAML, tqdm, tensorboardX, wandb, numba

Run `pip install -e torchlight`.

You could find more details in the `requirements.txt`, or use the command `pip install -r requirements.txt`.
注意在安装apex时需要
```shell
cd apex
python setup.py install
```
在训练Mix_former时可能出现有关于apex的报错，需要将报错位置
`if cached x.grad fn.next functions[1][0].variable is not x:`
改为
```
next functions available=False
if next functions_available and cached x.grad fn.next functions[1][0].variable is not x:
```

## 数据处理
```shell
cd UAV-SAR/data/uva_aug
python data_preprocess.py #需要修改好数据路径
python data_augmentation.py   #我们使用了该数据增强代码
```
注意路径

## Train
For example, these are commands for training Skeleton-Mixformer on v1. Please change the arguments `--config` and `--work-dir` to custom your training. If you want to training on v2, we have prepared the arguments in `./mixformer/config`.

1. We trained the model in k=1, k=2 and k=6. You could use the commands as:
```shell
cd ./mixformer

# k=1
python main.py --config ./config/uav_csv1/_1.yaml --work-dir <the save path of results>

# k=2
python main.py --config ./config/uav_csv1/_2.yaml --work-dir <the save path of results>

# k=6
python main.py --config ./config/uav_csv1/_6.yaml --work-dir <the save path of results>
```
 
## Test
我们使用与上述相同的方法更换test数据为B，经过上述数据处理得到MMVRAC_CSv2_B.npz（未进行数据增强）
然后使用指令
```shell
python main.py --config ./config/uav_csv2/config_name.yaml --work-dir save_path --phase 'test' --weights your_pt_path
#sach as python main.py --config ./config/uav_csv2/_6.yaml --work-dir /home/zjl_laoshi/xiaoke/UAV-SAR/mixformer/results/B/_6 --phase 'test' --weights /home/zjl_laoshi/xiaoke/UAV-SAR/mixformer/results/A/_6/runs-64-24640.pt
```
得到_1、_2、_6的预测结果

# Ensemble

## Ensemble Mix_GCN、Mix_Former和TEGCN

**1.** You can obtain the final classification accuracy of CSv1 by running the following code:
```shell
python Ensemble_B.py
```
注意：在运行上述指令前，需要该修改文件中的路径,并按默认值对应的结果依次更改new_test_r1_Score到new_test_r21_Score
我们的权重比自己设置的，没有自动搜素的过程

当然，我们的网盘中提供了我们的结果，在融合我们的结果前也需要修改成对应的路径

# Contact
在提交前，我们已经成功复现，如果在复现细节上有疑问，可以通过QQ或者邮件联系我们，谢谢！
QQ：3091166956
email:3091166956@qq.com

