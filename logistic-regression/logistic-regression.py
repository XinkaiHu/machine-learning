# Logistic Regression

import numpy as np
import os


def load_data(
    file: str,
    sep: str = ','
) -> tuple[np.matrix]:
    """
    从文件中加载数据。要求数据文件每行包含一个样例且每行最后一个数据为样例标签。
    input:
        file: 数据文件路径。
        sep: 文件中每个特征的分隔符。
    output:
        X: 每行表示一个样本、每列表示一个样本特征的矩阵。
        Y: 每行表示一个样本标签、列数为 1 的向量。
    """
    with open(
            file=file,
            mode='r') as f:
        data = np.matrix(data=[[
            np.float64(feature)
            for feature in sample.strip().split(sep=sep)]
            for sample in f.readlines()],
            dtype=np.float64)

    X = data[:, :-1]
    Y = data[:, -1]

    return X, Y


def make_multiclass(
    Y: np.matrix
) -> np.matrix:
    table = dict()
    for label in Y:
        if label not in table.values():
            table[len(table)] = label

    multiclass = np.matrix(data=[[
        np.float64(1.) if label == table[i] else np.float64(0.)
        for label in Y]
        for i in table.keys()],
        dtype=np.float64)

    return multiclass, table


def make_one_hot(
    Y: np.matrix
) -> np.matrix:
    """
    将标签值转化为独热码。
    input:
        Y: 标签值 shape(m, 1)
    output:
        one-hot: 独热码 shape(m, l)
    """
    return np.matrix(data=[
        np.eye(1, Y.max() + 1, y[0, 0], dtype=np.float64)[0, :]
        for y in Y],
        dtype=np.float64)


def normalization(
    X: np.matrix
) -> np.matrix:
    mu = np.mean(X, axis=1)
    sigma = np.std(X, axis=1)
    return (X - mu) / sigma


def load_parameters(
        file: str,
        sep: str = ','
) -> tuple[np.matrix]:
    """
    与 save_parameters 对应、加载文件中的参数。
    input:
        file: 保留参数的文件路径。
    output:
        文件中保留的 weights 和 bias。
    """
    with open(file=file, mode='r') as f:
        parameters = np.matrix([[
            np.float64(parameter)
            for parameter in classifier.strip().split(sep=sep)]
            for classifier in f.readlines()])
    weights = parameters[:, :-1]
    bias = parameters[:, -1]

    return weights, bias


def save_parameters(
    file: str,
    weights: np.matrix,
    bias: np.matrix,
    sep: str = ','
) -> None:
    """
    将参数保留到文件中。
    input:
        file: 保留参数的文件路径。
        weights: 同 predict 中的 weights。
        bias: 同 predict 中的 bias。
    """
    with open(file=file, mode='w') as f:
        line, row = weights.shape
        for i in range(line):
            for j in range(row):
                f.write('{}'.format(weights[i, j]) + sep)
            f.write('{}\n'.format(bias[i, 0]))


def sigmoid(
    z: np.matrix
) -> np.matrix:
    """
    Sigmoid(z) = 1 / (1 + exp(-z))。
    """
    return np.float64(1.) / (np.float64(1.) + np.exp(-z))


def predict(
    X: np.matrix,
    weights: np.matrix,
    bias: np.matrix,
    table: dict[np.matrix]
) -> np.matrix:
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
    m = X.shape[0]
    prediction = sigmoid(
        np.matmul(weights, X.T)
        + np.matmul(bias, np.ones(shape=(1, m), dtype=np.float64)))
    prediction /= np.sum(prediction, axis=0)
    # print(prediction)
    argmax_pred = np.argmax(prediction, axis=0)
    # print(np.matrix(
    #     data=[table[argmax_pred[0, i]][0, 0] for i in range(m)],
    #     dtype=np.float64))
    return np.matrix(
        data=[table[argmax_pred[0, i]][0, 0] for i in range(m)],
        dtype=np.float64)


def gradient_descent(
    X: np.matrix,
    Y: np.matrix,
    weights: np.matrix,
    bias: np.matrix,
    lr: np.float64
) -> None:
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
    m = X.shape[0]
    prediction = sigmoid(
        np.matmul(weights, X.T)
        + np.matmul(bias, np.ones(shape=(1, m), dtype=np.float64)))
    weights -= lr * np.matmul(prediction - Y, X) / m
    bias -= lr * np.matmul(
        prediction - Y, np.ones(shape=(m, 1), dtype=np.float64)) / m


def test(
    X: np.matrix,
    Y: np.matrix,
    weights: np.array,
    bias: np.matrix,
    table: dict[np.matrix]
) -> None:
    """
    根据给定的测试集和模型参数、测试模型的正确率。
    是 test 针对多分类问题的改进。
    input:
        X: 同 load_data 中的 X。
        Y: 同 load_data 中的 Y。
        weights: 同 multi_predict 中的 weights。
        bias: 同 multi_predict 中的 bias。
    """
    prediction = predict(X=X, weights=weights, bias=bias, table=table)
    m = X.shape[0]
    right = 0
    error = 0
    for i in range(m):
        if prediction[0, i] == Y[i, 0]:
            right += 1
        else:
            error += 1
    print("Right: {}, error: {}, right rate: {}".format(
        right, error, right / (right + error)))


melon_X_training, melon_Y_training = load_data(
    file=os.path.join(__file__, '..', 'dataset', 'melon_training.txt'))
melon_X_test, melon_Y_test = load_data(
    file=os.path.join(__file__, '..', 'dataset', 'melon_test.txt'))

melon_Y_training, melon_table = make_multiclass(Y=melon_Y_training)

melon_weights, melon_bias = load_parameters(
    file=os.path.join(__file__, '..', 'parameters', 'melon_parameters.txt'))


epoch = 1000
for i in range(epoch):
    gradient_descent(
        X=melon_X_training,
        Y=melon_Y_training,
        weights=melon_weights,
        bias=melon_bias,
        lr=0.5)

test(
    X=melon_X_test,
    Y=melon_Y_test,
    weights=melon_weights,
    bias=melon_bias,
    table=melon_table)

save_parameters(
    file=os.path.join(__file__, '..', 'parameters', 'melon_parameters.txt'),
    weights=melon_weights,
    bias=melon_bias)


iris_X_training, iris_Y_training = load_data(
    file=os.path.join(__file__, '..', 'dataset', 'iris_training.txt'))
iris_X_test, iris_Y_test = load_data(
    file=os.path.join(__file__, '..', 'dataset', 'iris_test.txt'))

iris_Y_training, iris_table = make_multiclass(Y=iris_Y_training)

iris_weights, iris_bias = load_parameters(
    file=os.path.join(__file__, '..', 'parameters', 'iris_parameters.txt'))

epoch = 1000
for i in range(epoch):
    gradient_descent(
        X=iris_X_training,
        Y=iris_Y_training,
        weights=iris_weights,
        bias=iris_bias,
        lr=0.1)

test(
    X=iris_X_test,
    Y=iris_Y_test,
    weights=iris_weights,
    bias=iris_bias,
    table=iris_table)

save_parameters(
    file=os.path.join(__file__, '..', 'parameters', 'iris_parameters.txt'),
    weights=iris_weights,
    bias=iris_bias)


cancer_X_training, cancer_Y_training = load_data(
    file=os.path.join(__file__, '..', 'dataset', 'cancer_training.txt'),
    sep='\t')
cancer_X_test, cancer_Y_test = load_data(
    file=os.path.join(__file__, '..', 'dataset', 'cancer_test.txt'),
    sep='\t')

cancer_X_training = normalization(cancer_X_training)
cancer_X_test = normalization(cancer_X_test)
cancer_Y_training, cancer_table = make_multiclass(Y=cancer_Y_training)

cancer_weights, cancer_bias = load_parameters(
    file=os.path.join(__file__, '..', 'parameters', 'cancer_parameters.txt'))
# cancer_weights = np.random.rand(2, 30)
# cancer_bias = np.random.rand(2, 1)

epoch = 10000
for i in range(epoch):
    gradient_descent(
        X=cancer_X_training,
        Y=cancer_Y_training,
        weights=cancer_weights,
        bias=cancer_bias,
        lr=0.05)

test(
    X=cancer_X_test,
    Y=cancer_Y_test,
    weights=cancer_weights,
    bias=cancer_bias,
    table=cancer_table)

save_parameters(
    file=os.path.join(__file__, '..', 'parameters', 'cancer_parameters.txt'),
    weights=cancer_weights,
    bias=cancer_bias)