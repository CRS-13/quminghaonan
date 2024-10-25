# 取名好难
这是我们的比赛文件

## Data preparation
Prepare the data according to dataset/dataset/read me.md

Your `data/` should be like this:
```
uav
___ data
    ___ test_data.npy
    ___ test_label.pkl
    ___ train_data.npy

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


