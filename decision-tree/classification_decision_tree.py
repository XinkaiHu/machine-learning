import numpy as np
import os


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


def load_data(file, sep, training_number, has_headers):
    """
    从文件中加载数据。要求数据文件每行包含一个样例且每行最后一个数据为样例标签。
    input:
        file: 数据文件路径。
        sep: 文件中每个特征的分隔符。
        training_number: 训练样本的个数、剩余的被分为测试样本。
        has_headers: 表示数据文件是否包含表头。
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

    X_training = X[:training_number, :]
    Y_training = Y[:training_number]
    X_test = X[training_number:, :]
    Y_test = Y[training_number:]

    if has_headers:
        return X_training, Y_training, X_test, Y_test, headers
    else:
        return X_training, Y_training, X_test, Y_test


def get_label_space(Y) -> np.matrix:
    """
    标记空间。
    """
    return np.unique(Y)


def information_entropy(Y: np.matrix) -> np.float64:
    """
    信息熵。
    """
    labels = get_label_space(Y=Y)
    m = len(Y)
    ent = np.float64(0.)
    for label in labels:
        p = np.count_nonzero(Y == label) / m
        ent -= p * np.log2(p)
    return ent


def get_most_label(Y: list):
    """
    返回标记集合中数目最多的类别。
    """
    label_space = get_label_space(Y=Y)
    most_label = label_space[0]
    most_label_count = np.count_nonzero(Y == most_label)
    for label in label_space[1:]:
        label_count = most_label_count = np.count_nonzero(Y == label)
        if label_count > most_label_count:
            most_label = label
            most_label_count = label_count
    return most_label


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
        best_entropy: 最佳划分时的信息熵。
        best_feature: 最佳划分特征。
        best_value: 最佳划分的值。
    """
    best_entropy = np.inf
    best_feature = -1
    best_value = -1
    d = X.shape[1]
    m = len(X)

    for i in range(d):
        sorted_index = np.argsort(X[:, i])
        for k in range(1, m):
            if X[sorted_index[k], i] != X[sorted_index[k - 1], i]:
                mid = (X[sorted_index[k], i] + X[sorted_index[k-1], i]) / 2
                X_left, X_right, Y_left, Y_right = split(X, Y, i, mid)

                p_left = len(X_left) / len(X)
                p_right = len(X_right) / len(X)

                ent = p_left * information_entropy(Y_left) + p_right * information_entropy(Y_right)
                if ent < best_entropy:
                    best_entropy = ent
                    best_feature = i
                    best_value = mid

    return best_entropy, best_feature, best_value


def build_tree(X, Y) -> tuple:
    """
    生成分类决策树。
    input:
        X: 样本集合。
        Y: 标记集合。
    output:
        决策树的根节点。
    """
    label_space = get_label_space(Y)
    if len(label_space) == 1:
        return Node(is_leaf=True, label=label_space[0])

    entropy, feature, value = get_partition(X=X, Y=Y)
    if entropy == 0:
        return Node(is_leaf=True, label=get_most_label(Y=Y))

    X_left, X_right, Y_left, Y_right = split(X=X, Y=Y, feature=feature, value=value)
    return Node(
        is_leaf=False,
        feature=feature,
        value=value,
        left=build_tree(X=X_left, Y=Y_left),
        right=build_tree(X=X_right, Y=Y_right))


def pretty_show(tree: Node, headers=lambda x: x, level: int=0):
    """
    打印分类决策树。
    """
    if tree.is_leaf:
        print(level * '\t' + 'label: {}'.format(tree.label))
    else:
        print(level * '\t' + 'partition feature: {}'.format(headers(tree.feature)))
        print(level * '\t' + 'If value <= {}'.format(tree.value))
        pretty_show(tree=tree.left, headers=headers, level=level+1)
        print(level * '\t' + 'If value > {}'.format(tree.value))
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


def test(prediction, Y):
    """
    根据对测试数据预测值和测试数据的真实值计算预测正确率。
    input:
        prediction: 预测值。
        Y: 真实值。
    """
    m = Y.shape[0]
    right = 0
    error = 0
    for k in range(m):
        if prediction[k] == Y[k]:
            right += 1
        else:
            error += 1
    print("Right: {}, error: {}, right rate: {}".format(
        right, error, 1 if m == 0 else right / (right + error)))


iris_X_training, iris_Y_training, iris_X_test, iris_Y_test = load_data(
    file=os.path.join(__file__, '..', 'dataset', 'iris.txt'),
    sep=',',
    training_number=100,
    has_headers=False)

iris_tree = build_tree(X=iris_X_training, Y=iris_Y_training)
print("========== iris tree ==========")
pretty_show(iris_tree)
iris_prediction = predict(iris_tree, iris_X_test)
test(prediction=iris_prediction, Y=iris_Y_test)

cancer_X_training, cancer_Y_training, cancer_X_test, cancer_Y_test = load_data(
    file=os.path.join(__file__, '..', 'dataset', 'cancer.txt'),
    sep='\t',
    training_number=450,
    has_headers=False)

cancer_tree = build_tree(X=cancer_X_training, Y=cancer_Y_training)
print("========== cancer tree ==========")
pretty_show(cancer_tree)
cancer_prediction = predict(cancer_tree, cancer_X_test)
test(cancer_prediction, cancer_Y_test)
