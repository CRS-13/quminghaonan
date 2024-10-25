import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str, default='/home/zjl_laoshi/xiaoke/dataset_xiaoke/eval/new_test_predictions.npy')

if __name__ == "__main__":

    args = parser.parse_args()

    # load label and pred
    # label =np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/zero_label_B.npy')
    label =np.load('/home/zjl_laoshi/xiaoke/dataset_xiaoke/data/test_A_label.npy')

    pred = np.load(args.pred_path).argmax(axis=1) #我们的npy文件为一维数据
    # pred = np.load(args.pred_path)

    correct = (pred == label).sum()

    total = len(label)

    print('Top1 Acc: {:.2f}%'.format(correct / total * 100))
