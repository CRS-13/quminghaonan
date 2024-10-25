import argparse
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--pred_path', type=str, default='/home/zjl_laoshi/xiaoke/UAV-SAR/infogcn/result/test/motion_6/epoch1_test_score.pkl')
parser.add_argument('--label_path', type=str, default='/home/zjl_laoshi/xiaoke/dataset_xiaoke/eval/test_label_A.npy')

if __name__ == "__main__":
    args = parser.parse_args()

    # load label from npy file
    label = np.load(args.label_path)

    # load predictions from pkl file
    with open(args.pred_path, 'rb') as f:
        pred = pickle.load(f)

    # Assuming pred is a 1D array or needs to be processed similarly
    # If pred is a probability array, you may need to use argmax, e.g.,
    # pred = np.array(pred).argmax(axis=1)

    correct = (pred == label).sum()
    total = len(label)

    print('Top1 Acc: {:.2f}%'.format(correct / total * 100))
