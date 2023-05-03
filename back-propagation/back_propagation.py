import numpy as np
import os


# m: sample
# d: input feature
# q: hidden feature
# l: output feature
# X: (m, d)
# Y: (m, l)
# hidden_weights: (q, d)
# hidden_bias: (q, 1)
# output_weights: (l, q)
# output_bias: (l, 1)


def normalization(
    X: np.matrix
) -> np.matrix:
    """
    将输入数据标准化。
    input:
        X: 输入数据 shape(m, d)
    output:
        标准化数据 shape(m, d)
    """
    m = X.shape[0]
    d = X.shape[1]
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    normalized_X = np.zeros_like(X)
    for i in range(d):
        for k in range(m):
            normalized_X[k, i] = (X[k, i] - mean[0, i]) / std[0, i]
    return normalized_X


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


def load_data(
    file: str,
    training_number: int,
    sep: str = ','
) -> tuple[np.matrix]:
    """
    从文件中加载数据。要求数据文件每行包含一个样例且每行最后一个数据为样例标签。
    input:
        file: 数据文件路径。
        training_number: 训练样本数。
        sep: 文件中每个特征的分隔符。
    output:
        X_training: 训练样本。
        Y_training: 训练样本标签。
        X_test: 测试样本。
        Y_test: 测试样本标签。
    """
    with open(file=file, mode='r') as f:
        data = np.matrix(data=[[
            np.float64(feature)
            for feature in sample.strip().split(sep=sep)]
            for sample in f.readlines()],
            dtype=np.float64)

    X = data[:, :-1]
    Y = np.matrix(data=[[int(y)] for y in data[:, -1]], dtype=np.int32)

    X = normalization(X=X)
    Y = make_one_hot(Y=Y)

    X_training = X[:training_number, :]
    Y_training = Y[:training_number, :]
    X_test = X[training_number:, :]
    Y_test = Y[training_number:, :]

    return X_training, Y_training, X_test, Y_test


def load_layer(
    file: str,
    sep: str = ','
) -> tuple[np.matrix]:
    """
    从文件中加载某一层网络的参数。
    input:
        file: 保存参数的文件路径。
        sep: 参数之间的分隔符。
    output:
        weights: 权重。
        bias: 阈值。
    """
    with open(file=file, mode='r') as f:
        parameters = np.matrix(data=[[
            np.float64(parameter)
            for parameter in classifier.strip().split(sep=sep)]
            for classifier in f.readlines()])
    weights = parameters[:, :-1]
    bias = parameters[:, -1]

    return weights, bias


def sigmoid(
    z: np.matrix
) -> np.matrix:
    """
    Sigmoid(z) = 1 / (1 + exp(-z))。
    """
    return np.float64(1.) / (np.float64(1.) + np.exp(-z))


def back_propagation(
    X: np.matrix,
    Y: np.matrix,
    hidden_weights: np.matrix,
    hidden_bias: np.matrix,
    output_weights: np.matrix,
    output_bias: np.matrix,
    lr: np.float64
) -> tuple[np.matrix]:
    """
    反向传播更新参数。
    input:
        X: 输入样本 shape(m, d)
        Y: 样本标签 shape(m, l)
        hidden_weights: 隐藏层权重 shape(q, d)
        hidden_bias: 隐藏层阈值 shape(q, 1)
        output_weights: 输出层权重 shape(l, q)
        output_bias: 输出层阈值 shape(l, 1)
        lr: 学习率
    """
    m = X.shape[0]
    d = X.shape[1]
    q = hidden_weights.shape[0]
    l = output_weights.shape[0]

    alpha = np.zeros(shape=(q, 1), dtype=np.float64)
    b = np.zeros(shape=(q, 1), dtype=np.float64)
    beta = np.zeros(shape=(l, 1), dtype=np.float64)
    y_hat = np.zeros(shape=(l, 1), dtype=np.float64)

    g = np.zeros(shape=(l, 1), dtype=np.float64)
    e = np.zeros(shape=(q, 1), dtype=np.float64)

    d_w = np.zeros(shape=(l, q), dtype=np.float64)
    d_theta = np.zeros(shape=(l, 1), dtype=np.float64)
    d_v = np.zeros(shape=(q, d), dtype=np.float64)
    d_gamma = np.zeros(shape=(q, 1), dtype=np.float64)

    for k in range(m):
        for h in range(q):
            sum_tmp = 0
            for i in range(d):
                sum_tmp += hidden_weights[h, i] * X[k, i]
            alpha[h, 0] = sum_tmp

        b = sigmoid(alpha - hidden_bias)

        for j in range(l):
            sum_tmp = 0
            for h in range(q):
                sum_tmp += output_weights[j, h] * b[h, 0]
            beta[j, 0] = sum_tmp

        y_hat = sigmoid(beta - output_bias)

        for j in range(l):
            g[j] = y_hat[j, 0] * (1 - y_hat[j, 0]) * (Y[k, j] - y_hat[j, 0])

        for h in range(q):
            sum_tmp = 0
            for j in range(l):
                sum_tmp += output_weights[j, h] * g[j, 0]
            e[h] = b[h, 0] * (1 - b[h, 0]) * sum_tmp

        for h in range(q):
            for j in range(l):
                d_w[j, h] = lr * g[j, 0] * b[h, 0]

        for j in range(l):
            d_theta[j, 0] = -lr * g[j, 0]

        for i in range(d):
            for h in range(q):
                d_v[h, i] = lr * e[h, 0] * X[k, i]

        for h in range(q):
            d_gamma[h, 0] = -lr * e[h, 0]

        output_weights += d_w
        output_bias += d_theta
        hidden_weights += d_v
        hidden_bias += d_gamma


def predict(
    X: np.matrix,
    hidden_weights: np.matrix,
    hidden_bias: np.matrix,
    output_weights: np.matrix,
    output_bias: np.matrix
) -> np.matrix:
    """
    根据权重和样本特征预测标签。
    input:
        X: 输入样本 shape(m, d)
        hidden_weights: 隐藏层权重 shape(q, d)
        hidden_bias: 隐藏层阈值 shape(q, 1)
        output_weights: 输出层权重 shape(l, q)
        output_bias: 输出层阈值 shape(l, 1)
    output:
        prediction: 标签的预测值 shape(m, l)
    """
    m = X.shape[0]
    d = X.shape[1]
    q = hidden_weights.shape[0]
    l = output_weights.shape[0]

    alpha = np.zeros(shape=(q, 1), dtype=np.float64)
    b = np.zeros(shape=(q, 1), dtype=np.float64)
    beta = np.zeros(shape=(l, 1), dtype=np.float64)
    y_hat = np.zeros(shape=(l, 1), dtype=np.float64)

    prediction = np.zeros(shape=(m, l), dtype=np.float64)

    for k in range(m):
        for h in range(q):
            sum_tmp = 0
            for i in range(d):
                sum_tmp += hidden_weights[h, i] * X[k, i]
            alpha[h, 0] = sum_tmp

        b = sigmoid(alpha - hidden_bias)

        for j in range(l):
            sum_tmp = 0
            for h in range(q):
                sum_tmp += output_weights[j, h] * b[h, 0]
            beta[j, 0] = sum_tmp

        y_hat = sigmoid(beta - output_bias)

        for j in range(l):
            prediction[k, j] = y_hat[j, 0]

    return prediction


def MSE(
    prediction: np.matrix,
    Y: np.matrix
) -> np.float64:
    """
    计算预测值与真实值的均方误差。
    input:
        prediction: 预测值 shape(m, l)
        Y: 真实值 shape(m, l)
    output:
        均方误差值。
    """
    m = Y.shape[0]
    l = Y.shape[1]
    return np.mean(np.matrix(data=[0.5 * np.sum([
        (prediction[k, j] - Y[k, j]) ** 2
        for j in range(l)])
        for k in range(m)], dtype=np.float64))


def test(
    prediction: np.matrix,
    Y: np.matrix
) -> None:
    """
    根据给定的测试集和模型参数、测试模型的正确率。
    input:
        prediction: 标签的预测值 shape(1, m)
        Y: 标签的实际值 shape(m, 1)
    """
    m = prediction.shape[0]
    print("MSE: {}".format(
        MSE(prediction=prediction, Y=Y)))

    right = 0
    error = 0
    for k in range(m):
        if np.argmax(prediction[k, :]) == np.argmax(Y[k, :]):
            right += 1
        else:
            error += 1
    print("Right: {}, error: {}, right rate: {}".format(
        right, error, 1 if m == 0 else right / (right + error)))


def save_layer(
    file: str,
    weights: np.matrix,
    bias: np.matrix,
    sep: str = ','
) -> None:
    """
    保存某一层网络的参数。
    input:
        file: 保存参数的文件路径。
        weights: 权重。
        bias: 阈值。
        sep: 参数之间的分隔符。
    """
    line, row = weights.shape
    with open(file=file, mode='w') as f:
        for i in range(line):
            for j in range(row):
                f.write('{}'.format(weights[i, j]) + sep)
            f.write('{}\n'.format(bias[i, 0]))


iris_X_training, iris_Y_training, iris_X_test, iris_Y_test = load_data(
    file=os.path.join(__file__, '..', 'dataset', 'iris.txt'),
    training_number=120)

iris_hidden_weights, iris_hidden_bias = load_layer(
    file=os.path.join(__file__, '..', 'parameters', 'iris_hidden_parameters.txt'))
iris_output_weights, iris_output_bias = load_layer(
    file=os.path.join(__file__, '..', 'parameters', 'iris_output_parameters.txt'))

epoch = 100
for i in range(epoch):
    back_propagation(
        X=iris_X_training,
        Y=iris_Y_training,
        hidden_weights=iris_hidden_weights,
        hidden_bias=iris_hidden_bias,
        output_weights=iris_output_weights,
        output_bias=iris_output_bias,
        lr=0.05)

iris_prediction = predict(
    X=iris_X_test,
    hidden_weights=iris_hidden_weights,
    hidden_bias=iris_hidden_bias,
    output_weights=iris_output_weights,
    output_bias=iris_output_bias)

print('iris prediction:\n{}\n'.format(np.argmax(iris_prediction, axis=1)))
print('iris Y test\n{}\n'.format(np.argmax(iris_Y_test, axis=1).T))

save_layer(
    file=os.path.join(__file__, '..', 'parameters',
                      'iris_hidden_parameters.txt'),
    weights=iris_hidden_weights,
    bias=iris_hidden_bias)

save_layer(
    file=os.path.join(__file__, '..', 'parameters',
                      'iris_output_parameters.txt'),
    weights=iris_output_weights,
    bias=iris_output_bias)

test(
    prediction=iris_prediction,
    Y=iris_Y_test)


cancer_X_training, cancer_Y_training, cancer_X_test, cancer_Y_test = load_data(
    file=os.path.join(__file__, '..', 'dataset', 'cancer.txt'),
    training_number=450,
    sep='\t')

cancer_hidden_weights, cancer_hidden_bias = np.random.rand(
    200, 30), np.random.rand(200, 1)
cancer_output_weights, cancer_output_bias = np.random.rand(
    2, 200), np.random.rand(2, 1)

epoch = 10
for i in range(epoch):
    back_propagation(
        X=cancer_X_training,
        Y=cancer_Y_training,
        hidden_weights=cancer_hidden_weights,
        hidden_bias=cancer_hidden_bias,
        output_weights=cancer_output_weights,
        output_bias=cancer_output_bias,
        lr=0.01)

cancer_prediction = predict(
    X=cancer_X_test,
    hidden_weights=cancer_hidden_weights,
    hidden_bias=cancer_hidden_bias,
    output_weights=cancer_output_weights,
    output_bias=cancer_output_bias)

print('cancer prediction:\n{}\n'.format(np.argmax(cancer_prediction, axis=1)))
print('cancer Y test:\n{}\n'.format(np.argmax(cancer_Y_test, axis=1).T))

save_layer(
    file=os.path.join(__file__, '..', 'parameters',
                      'cancer_hidden_parameters.txt'),
    weights=cancer_hidden_weights,
    bias=cancer_hidden_bias)

save_layer(
    file=os.path.join(__file__, '..', 'parameters',
                      'cancer_output_parameters.txt'),
    weights=cancer_output_weights,
    bias=cancer_output_bias)

test(
    prediction=cancer_prediction,
    Y=cancer_Y_test)
