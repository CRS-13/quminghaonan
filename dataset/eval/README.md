# 测试脚本

## 文件说明

1. 标签文件`label.npy`
2. 测试脚本`eval.py`
3. 置信度文件`pred.npy`

**其中标签文件与测试脚本是不需要变动的文件**

## 使用说明

将标签文件`label.npy`和测试脚本`eval.py`放在同一目录下，运行`python eval.py --pred_path PATH`即可

## 输入说明

`--pred_path PATH`

其中`PATH`为置信度文件`pred.npy`的路径，例如`test/pred.npy`等
必须完整包括文件名

## 输出说明

输出类似以下
`Top1 Acc: 55.16%`

其中`55.16`为最终排名依据