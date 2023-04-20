# Logistic Regression

import numpy as np
import os


def load_data(file: str, sep: str = ' ') -> tuple[np.array]:
    """
    从文件中加载数据。要求数据文件每行包含一个样例且每行最后一个数据为样例标签。
    input:
        file: 数据文件路径。
        sep: 文件中每个特征的分隔符。
    output:
        X: 每行表示一个样本、每列表示一个样本特征的矩阵。
        Y: 每行表示一个样本标签、列数为 1 的向量。
    """
    with open(file=file, mode='r') as f:
        data = np.array([[
            np.float64(feature)
            for feature in sample.strip().split(sep=sep)]
            for sample in f.readlines()])

    X = data[:, :-1]
    Y = data[:, -1]

    return X, Y


def sigmoid(z: np.array) -> np.array:
    """
    Sigmoid(z) = 1 / (1 + exp(-z))。
    """
    return np.float64(1.) / (np.float64(1.) + np.exp(-z))


def predict(X: np.matrix, weights: np.array, bias: np.array) -> np.array:
    """
    根据样本和权重预测样本的标签。
    P{y = 1 | X} = sigmoid(w^T x + b)。
    input:
        X: 同 load_data 中的 X。
        weights: 行数为样本特征数、列数为 1 的权重向量。即公式中的 w。
        bias: 即公式中的 b。
    output:
        与 load_data 中 Y 形状相同、对 Y 的预测值。
    """
    positive = sigmoid(np.dot(X, weights) + bias)
    negative = 1 - positive
    return np.argmax(np.array([negative, positive]), axis=0)


def multi_predict(X: np.matrix, weights: np.matrix, bias: np.array) -> np.array:
    """
    根据样本和权重预测样本的标签。
    是 predict 针对多分类问题的改进。
    假设共有 C 个类别、训练 C 个 logistic regression 模型、
    预测时计算样本属于每个类的概率、取概率最大者作为样本的概率。
    input:
        X: 同 predict 中的 X。
        weights: 与 predict 中的不同、此处 weights 有 C 行、每行对应 predict 中
            一个 logistic regression 模型的 weights。
        bias: 与 predict 中的不同。此处 bias 为 C 行 1 列的向量。
    output:
        与 load_data 中 Y 形状相同、对 Y 的预测值。
    """
    pred = sigmoid(np.matmul(X, weights.T) +
                   np.matmul(np.ones((X.shape[0], 1), dtype=np.float64), bias.T))
    return np.argmax(pred, axis=1)


def gradient_descent(X: np.matrix, Y: np.array, weights: np.array, bias: np.array, lr: np.float64) -> None:
    """
    采用梯度下降法优化损失函数。损失函数采用交叉熵损失、其公式为
        loss = -y ln y - (1 - y) ln(1 - y)。
    对应的参数更新公式为
        w_j := w_j - alpha / m * sum_i ((yhat_i - y_i) * x_i_j)
        b := b - alpha / m * sum_i (yhat_i - y_i)
    input:
        X: 同 load_data 中的 X。
        Y: 同 load_data 中的 Y。
        weights: 同 predict 中的 weights。
        bias: 同 predict 中的 bias。
        lr: 学习率。即公式中的 alpha
    """
    predictions = sigmoid(np.dot(X, weights) + bias)
    weights -= lr * np.dot(predictions - Y, X) / len(Y)
    bias -= lr * np.dot(np.ones((1, len(Y)), dtype=np.float64),
                        predictions - Y) / len(Y)


def save_parameters(file: str, weights: np.array, bias: np.array) -> None:
    """
    将参数保留到文件中。
    input:
        file: 保留参数的文件路径。
        weights: 同 predict 中的 weights。
        bias: 同 predict 中的 bias。
    """
    with open(file=file, mode='w') as f:
        for w in weights:
            f.write('{},'.format(w))
        f.write('{}'.format(bias[0]))


def load_parameters(file: str) -> tuple[np.array]:
    """
    与 save_parameters 对应、加载文件中的参数。
    input:
        file: 保留参数的文件路径。
    output:
        文件中保留的 weights 和 bias。
    """
    with open(file=file, mode='r') as f:
        para = f.readline().strip().split(',')
        weights = np.array(para[:-1], dtype=np.float64)
        bias = np.array(para[-1:], dtype=np.float64)
    return weights, bias


def test(X: np.matrix, Y: np.array, weights: np.array, bias: np.array) -> None:
    """
    根据给定的测试集和模型参数、测试模型的正确率。
    input:
        X: 同 load_data 中的 X。
        Y: 同 load_data 中的 Y。
        weights: 同 predict 中的 weights。
        bias: 同 predict 中的 bias。
    """
    predictions = predict(X=X, weights=weights, bias=bias)
    right = 0
    error = 0
    for i in range(len(Y)):
        if predictions[i] == Y[i]:
            right += 1
        else:
            error += 1
    print("Right: {}, error: {}, right rate: {}".format(
        right, error, right / (right + error)))


def multi_test(X: np.matrix, Y: np.array, weights: np.array, bias: np.array) -> None:
    """
    根据给定的测试集和模型参数、测试模型的正确率。
    是 test 针对多分类问题的改进。
    input:
        X: 同 load_data 中的 X。
        Y: 同 load_data 中的 Y。
        weights: 同 multi_predict 中的 weights。
        bias: 同 multi_predict 中的 bias。
    """
    predictions = multi_predict(X=X, weights=weights, bias=bias)
    right = 0
    error = 0
    for i in range(len(Y)):
        if predictions[i] == Y[i]:
            right += 1
        else:
            error += 1
    print("Right: {}, error: {}, right rate: {}".format(
        right, error, right / (right + error)))


# 对西瓜数据集训练 weights 和 bias。
X_training, Y_training = load_data(os.path.join(
    __file__, '..', 'dataset', 'melon_training.txt'), ',')
X_test, Y_test = load_data(os.path.join(
    __file__, '..', 'dataset', 'melon_test.txt'), ',')
melon_weights, melon_bias = load_parameters(os.path.join(__file__, '..', 'parameters.txt')) if os.path.exists(
    os.path.join(__file__, '..', 'parameters.txt')) else np.random.rand(2), np.random.rand(1)

lr = 0.05
epoch = 10000
for _ in range(epoch):
    gradient_descent(X=X_training, Y=Y_training,
                     weights=melon_weights, bias=melon_bias, lr=lr)
save_parameters(os.path.join(__file__, '..', 'parameters.txt'),
                weights=melon_weights, bias=melon_bias)


print('---- My Logistic Regression ----\nWeights: \n{}\nBias: \n{}'.format(melon_weights, melon_bias))
test(X=X_test, Y=Y_test, weights=melon_weights, bias=melon_bias)


# 对鸢尾花数据集分类。
iris_X_test, iris_Y_test = load_data(file=os.path.join(
    __file__, '..', 'dataset', 'iris_test.txt'), sep=',')
iris_Y_test_0 = np.array(
    [np.float64(1.) if y == 0. else np.float64(0.) for y in iris_Y_test])
iris_Y_test_1 = np.array(
    [np.float64(1.) if y == 1. else np.float64(0.) for y in iris_Y_test])
iris_Y_test_2 = np.array(
    [np.float64(1.) if y == 2. else np.float64(0.) for y in iris_Y_test])

iris_X_training, iris_Y_training = load_data(
    file=os.path.join(__file__, '..', 'dataset', 'iris_training.txt'), sep=',')
iris_Y_training_0 = np.array(
    [np.float64(1.) if y == 0. else np.float64(0.) for y in iris_Y_training])
iris_Y_training_1 = np.array(
    [np.float64(1.) if y == 1. else np.float64(0.) for y in iris_Y_training])
iris_Y_training_2 = np.array(
    [np.float64(1.) if y == 2. else np.float64(0.) for y in iris_Y_training])

weights_0, bias_0 = load_parameters(os.path.join(__file__, '..', 'parameter_0.txt')) if os.path.exists(
    os.path.join(__file__, '..', 'parameter_0.txt')) else np.random.rand(4), np.random.rand(1)
weights_1, bias_1 = load_parameters(os.path.join(__file__, '..', 'parameter_1.txt')) if os.path.exists(
    os.path.join(__file__, '..', 'parameter_1.txt')) else np.random.rand(4), np.random.rand(1)
weights_2, bias_2 = load_parameters(os.path.join(__file__, '..', 'parameter_2.txt')) if os.path.exists(
    os.path.join(__file__, '..', 'parameter_2.txt')) else np.random.rand(4), np.random.rand(1)


lr = 0.05
epoch = 10000
for _ in range(epoch):
    gradient_descent(X=iris_X_training, Y=iris_Y_training_0,
                     weights=weights_0, bias=bias_0, lr=lr)
    gradient_descent(X=iris_X_training, Y=iris_Y_training_1,
                     weights=weights_1, bias=bias_1, lr=lr)
    gradient_descent(X=iris_X_training, Y=iris_Y_training_2,
                     weights=weights_2, bias=bias_2, lr=lr)

save_parameters(os.path.join(__file__, '..', 'parameters_0.txt'),
                weights=weights_0, bias=bias_0)
save_parameters(os.path.join(__file__, '..', 'parameters_1.txt'),
                weights=weights_1, bias=bias_1)
save_parameters(os.path.join(__file__, '..', 'parameters_2.txt'),
                weights=weights_2, bias=bias_2)

weights = np.matrix([weights_0, weights_1, weights_2])
bias = np.matrix([bias_0, bias_1, bias_2])
print('---- My Multi-Class Logistic Regression ----\nWeights: \n{}\nBias: \n{}'.format(weights, bias))
print('In training set:')
multi_test(X=iris_X_training, Y=iris_Y_training, weights=weights, bias=bias)
print('In test set:')
multi_test(X=iris_X_test, Y=iris_Y_test, weights=weights, bias=bias)
