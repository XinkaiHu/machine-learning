import numpy as np
import os
import matplotlib.pyplot as plt


class Node:
    def __init__(
        self, is_leaf, 
        feature=None,
        value=None,
        label=None,
        left=None,
        right=None
    ) -> None:
        self.is_leaf = is_leaf
        self.feature = feature
        self.value = value
        self.label = label
        self.left = left
        self.right = right


def load_data(file, sep, training_number, has_headers: bool, shuffle: bool=True):
    """
    从文件中加载数据。要求数据文件每行包含一个样例且每行最后一个数据为样例标签。
    input:
        file: 数据文件路径。
        sep: 文件中每个特征的分隔符。
        training_number: 训练样本的个数、剩余的被分为测试样本。
        has_headers: 表示数据文件是否包含表头。
        shuffle: 表示是否打乱数据集。默认为 True。
    output:
        X: 每行表示一个样本、每列表示一个样本特征的矩阵。
        Y: 每行表示一个样本标签、列数为 1 的向量。
        headers: 表头。
    """
    with open(
            file=file,
            mode='r') as f:
        if has_headers:
            headers = f.readline().strip().split(sep=sep)
        data = np.array([[np.float64(feature)
            for feature in sample.strip().split(sep=sep)]
            for sample in f.readlines()],
            dtype=np.float64)

    X = data[:, :-1]
    Y = data[:, -1]

    if shuffle:
        np.random.shuffle(data)

    X_training = X[:training_number, :]
    Y_training = Y[:training_number]
    X_test = X[training_number:, :]
    Y_test = Y[training_number:]

    if not has_headers:
        return X_training, Y_training, X_test, Y_test
    else:
        return X_training, Y_training, X_test, Y_test, headers


def MSE(prediction, Y):
    """
    平方误差。用于确定最优划分点、评价回归树的性能。
    """
    return np.mean(np.square(prediction - Y))


def split(X, Y, feature, value):
    """
    把数据集根据特征 feature 的取值进行划分、
    分为不大于 value 和大于 value 的两部分。
    input:
        X: 样本集合。
        Y: 标记集合。
        feature: 进行划分依赖的特征。
        value: 对应特征上的分界值。
    """
    left = (X[:,feature] <= value)
    right = (X[:,feature] > value)
    return X[left], X[right], Y[left], Y[right]


def get_partition(X, Y):
    """
    计算最佳划分特征及最佳划分点。
    input:
        X: 样本集合。
        Y: 标记集合。
    output:
        best_feature: 最佳划分特征。
        best_value: 最佳划分的值。
    """
    best_loss = np.inf
    best_feature = None
    best_value = None
    d = X.shape[1]
    m = len(X)

    for i in range(d):
        sorted_index = np.argsort(X[:, i])
        for k in range(1, m):
            if X[sorted_index[k], i] != X[sorted_index[k - 1], i]:
                _, _, Y_left, Y_right = split(X, Y, i, X[sorted_index[k - 1], i])

                assert(len(Y_left) != 0 and len(Y_right) != 0)
                prediction_left = np.mean(Y_left)
                prediction_right = np.mean(Y_right)
                loss = MSE(prediction_left, Y_left) + MSE(prediction_right, Y_right)
                if loss < best_loss:
                    best_loss = loss
                    best_feature = i
                    best_value = X[sorted_index[k - 1], i]

    return best_feature, best_value


def build_tree(X, Y, level=0) -> tuple:
    """
    D: 训练集
    A: 属性集
    X: 训练集样本
    Y: 训练集样本标签
    """
    feature, value = get_partition(X=X, Y=Y)
    if feature is None:
        return Node(is_leaf=True, label=np.mean(Y))
    X_left, X_right, Y_left, Y_right = split(X=X, Y=Y, feature=feature, value=value)
    if len(Y_left) == 0:
        return Node(is_leaf=True, label=np.mean(Y_right))
    elif len(Y_right) == 0:
        return Node(is_leaf=True, label=np.mean(Y_left))

    return Node(
        is_leaf=False,
        feature=feature,
        value=value,
        left=build_tree(X=X_left, Y=Y_left, level=level+1),
        right=build_tree(X=X_right, Y=Y_right, level=level+1))


def pretty_show(tree: Node, headers=lambda x: x, level: int=0):
    """
    打印回归决策树。（不推荐使用）
    """
    if tree.is_leaf:
        print(level * '  ' + 'label: {}'.format(tree.label))
    else:
        print(level * '  ' + 'partition feature: {}'.format(headers(tree.feature)))
        print(level * '  ' + 'If value <= {}'.format(tree.value))
        pretty_show(tree=tree.left, headers=headers, level=level+1)
        print(level * '  ' + 'If value > {}'.format(tree.value))
        pretty_show(tree=tree.right, headers=headers, level=level+1)


def predict(tree: Node, X: np.matrix):
    """
    根据输入数据进行预测。
    input:
        tree: 决策树的根节点。
        X: 测试样本集合。
    output:
        对测试样本的预测值。
    """
    def __predict(__tree: Node, __X):
        if __tree.is_leaf:
            return __tree.label
        
        __feature = __tree.feature
        __value = __tree.value
        if __X[__feature] <= __value:
            return __predict(__tree.left, __X)
        else:
            return __predict(__tree.right, __X)

    return np.array([__predict(tree, X[k, :]) for k in range(X.shape[0])], dtype=np.float64)


mark_X_training, mark_Y_training, mark_X_test, mark_Y_test, mark_headers = load_data(
    file=os.path.join(__file__, '..', 'dataset', 'Student_Marks.csv'),
    sep=',',
    training_number=60,
    has_headers=True)

mark_tree = build_tree(X=mark_X_training, Y=mark_Y_training)
mark_prediction = predict(mark_tree, mark_X_test)

plt.figure()
plt.title("Real Label and Prediction in MARK Dataset")
plt.scatter(mark_Y_test, mark_prediction)
plt.xlabel("Real Label")
plt.ylabel("Prediction")
plt.xlim(0, 60)
plt.ylim(0, 60)
plt.show()


machine_X_training, machine_Y_training, machine_X_test, machine_Y_test = load_data(
    file=os.path.join(__file__, '..', 'dataset', 'machine.csv'),
    sep=',',
    training_number=150,
    has_headers=False
)

machine_tree = build_tree(X=machine_X_training, Y=machine_Y_training)
machine_prediction = predict(machine_tree, machine_X_test)

plt.figure()
plt.title("Real Label and Prediction in MACHINE Dataset")
plt.scatter(machine_Y_test, machine_prediction)
plt.xlabel("Real Label")
plt.ylabel("Prediction")
plt.xlim(0, 200)
plt.ylim(0, 200)
plt.show()
