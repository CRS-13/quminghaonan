import argparse
import pickle
import numpy as np
from tqdm import tqdm

def predict_with_weights(weights, r_values):
    weighted_sum = np.zeros_like(r_values[0])
    for r_val, weight in zip(r_values, weights):
        weighted_sum += r_val * weight
    return weighted_sum

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--new_test_r1_Score', default = '/home/zjl_laoshi/xiaoke/Top/Model_inference/Mix_Former/output/result/skmixf__V1_J/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r2_Score', default = '/home/zjl_laoshi/xiaoke/Top/Model_inference/Mix_Former/output/result/skmixf__V1_B/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r3_Score', default = '/home/zjl_laoshi/xiaoke/Top/Model_inference/Mix_Former/output/result/skmixf__V1_JM/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r4_Score', default = '/home/zjl_laoshi/xiaoke/Top/Model_inference/Mix_Former/output/result/skmixf__V1_BM/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r5_Score', default = '/home/zjl_laoshi/xiaoke/Top/Model_inference/Mix_Former/output/result/skmixf__V1_k2/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r6_Score', default = '/home/zjl_laoshi/xiaoke/Top/Model_inference/Mix_Former/output/result/skmixf__V1_k2M/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r7_Score', default = '/home/zjl_laoshi/xiaoke/Top/Model_inference/Mix_GCN/output/result/ctrgcn_V1_J/epoch1_test_score.pkl')  ##**
    parser.add_argument('--new_test_r8_Score', default = '/home/zjl_laoshi/xiaoke/Top/Model_inference/Mix_GCN/output/result/ctrgcn_V1_B/epoch1_test_score.pkl')  ##**
    parser.add_argument('--new_test_r9_Score', default = '/home/zjl_laoshi/xiaoke/Top/Model_inference/Mix_GCN/output/result/ctrgcn_V1_J_3D/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r10_Score', default = '/home/zjl_laoshi/xiaoke/Top/Model_inference/Mix_GCN/output/result/ctrgcn_V1_B_3D/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r11_Score', default = '/home/zjl_laoshi/xiaoke/Top/Model_inference/Mix_GCN/output/result/tdgcn_V1_J/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r12_Score', default = '/home/zjl_laoshi/xiaoke/Top/Model_inference/Mix_GCN/output/result/tdgcn_V1_B/epoch1_test_score.pkl')
    parser.add_argument('--new_test_r13_Score', default = '/home/zjl_laoshi/xiaoke/Top/Model_inference/Mix_GCN/output/result/mstgcn_V1_J/epoch1_test_score.pkl')  ##**
    parser.add_argument('--new_test_r14_Score', default = '/home/zjl_laoshi/xiaoke/Top/Model_inference/Mix_GCN/output/result/mstgcn_V1_B/epoch1_test_score.pkl')  ##**    arg = parser.parse_args()
    arg = parser.parse_args()

    # Load new test data scores
    scores = []
    for i in range(1, 15):
        with open(getattr(arg, f'new_test_r{i}_Score'), 'rb') as f:
            scores.append(list(pickle.load(f).items()))

    accuracies = [0.7, 0.7, 0.2, 0.2, 0.2, 0.2, 0.7, 0.7, 0.2, 0.2, 0.7, 0.7, 0.7, 0.7, 0.7]  

    # Normalize accuracies to sum to 1 for weights
    weights = np.array(accuracies) / sum(accuracies)

    # Apply weights on the new test set
    predictions_new = []
    for i in range(len(scores[0])):
#        print(len(scores[1][1]))
        r_values_new = [scores[j][i][1] for j in range(len(scores))]
        prediction = predict_with_weights(weights, r_values_new)
        predictions_new.append(prediction)

    # Save the new test set predictions
    np.save('new_test_predictions.npy', predictions_new)
