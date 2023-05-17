import math
import os
import numpy as np


def load_data(file: str, sep: str, has_headers: bool, shuffle: bool=True):
    """
    加载字符串数据、用于分类决策树。
    input:
        file: 数据文件路径。
        sep: 文件分隔符。
        has_headers: 表示数据文件是否包含表头。
    output:
        X: 每行表示一个样本、每列表示一个样本特征的矩阵。
        Y: 每行表示一个样本标签、列数为 1 的向量。
        headers: 表头。
        shuffle: 表示是否打乱数据集。默认为 True。
    """
    with open(
            file=file,
            mode='r', encoding="utf-8") as f:
        if has_headers:
            headers = f.readline().strip().split(sep=sep)
        data = [sample.strip().split(sep=sep)
            for sample in f.readlines()]
    
    if shuffle:
        np.random.shuffle(data)

    X = [sample[:-1] for sample in data]
    Y = [sample[-1] for sample in data]

    if has_headers:
        return X, Y, headers
    else:
        return X, Y


def get_label_space(Y: list) -> list:
    """
    标记空间。
    """
    return list(set(Y))


def get_sample_space(X: list) -> dict:
    """
    样本空间。
    """
    m = len(X)
    d = len(X[0])
    features = dict()
    for i in range(d):
        features_i = []
        for k in range(m):
            if X[k][i] not in features_i:
                features_i.append(X[k][i])
        features[i] = features_i
    return features


def information_entropy(Y: list) -> float:
    """
    信息熵。
    """
    labels = get_label_space(Y=Y)
    m = len(Y)
    ent = 0
    for label in labels:
        p = Y.count(label) / m
        ent -= p * math.log(p, 2)
    return ent


def information_gain(X, Y) -> list:
    """
    信息增益。
    """
    m = len(X)
    d = len(X[0])
    overall_entropy = information_entropy(Y=Y)
    sample_space = get_sample_space(X=X)
    gain = d * [overall_entropy]
    for i in range(d):
        for v in sample_space[i]:
            y = []
            for k in range(m):
                if X[k][i] == v:
                    y.append(Y[k])
            gain[i] -= len(y) * information_entropy(Y=y) / m
    return gain


def gain_ratio(X, Y) -> list:
    """
    信息增益率。
    """
    m = len(X)
    d = len(X[0])
    intrinsic_value = d * [0]
    overall_entropy = information_entropy(Y=Y)
    sample_space = get_sample_space(X=X)
    gain = d * [overall_entropy]
    for i in range(d):
        for v in sample_space[i]:
            y = []
            for k in range(m):
                if X[k][i] == v:
                    y.append(Y[k])
            intrinsic_value[i] -= len(y) * math.log2(len(y) / m) / m
            gain[i] -= len(y) * information_entropy(Y=y) / m
        gain[i] /= intrinsic_value[i] + 1e-8
    return gain


def gini(Y):
    """
    基尼值。
    """
    labels = get_label_space(Y=Y)
    m = len(Y)
    gini_value = 1
    for label in labels:
        p = Y.count(label) / m
        gini_value -= p ** 2
    return gini_value


def gini_index(X, Y):
    """
    基尼指数。
    """
    m = len(X)
    d = len(X[0])
    gini_index_value = d * [0]
    sample_space = get_sample_space(X=X)
    for i in range(d):
        for v in sample_space[i]:
            x, y = [], []
            for k in range(m):
                if X[k][i] == v:
                    x.append(X[k])
                    y.append(Y[k])
            gini_index_value[i] -= len(y) * gini(Y=y) / m
    return gini_index_value


def get_partition_feature(gain, disabled_features) -> int:
    """
    选择最优划分属性。
    input:
        gain: 信息增益、或其他划分准则。
        disabled_features: 辅助变量、记录已经使用过的划分属性。
    output:
        max_gain_index: 最优划分属性。
    """
    d = len(gain)
    max_gain_index = 0
    while max_gain_index in disabled_features:
        max_gain_index += 1
    for i in range(max_gain_index + 1, d):
        if i not in disabled_features and gain[i] > gain[max_gain_index]:
            max_gain_index = i
    return max_gain_index


def get_most_label(Y: list):
    """
    返回标记集合中数目最多的类别。
    """
    label_space = get_label_space(Y=Y)
    most_label = label_space[0]
    most_label_count = Y.count(most_label)
    for label in label_space[1:]:
        label_count = Y.count(label)
        if label_count > most_label_count:
            most_label = label
            most_label_count = label_count
    return most_label


# tree: (class_or_feature, {feature_value: subtree})
# class: Any, feature: int, feature_value: Any
def build_tree(X, Y, disabled_features = []) -> tuple:
    """
    生成分类决策树。
    input:
        X: 样本集合。
        Y: 标记集合。
        disabled_features: 辅助变量、记录已经使用过的划分属性。
    output:
        形如 tree = (class_or_feature, {feature_value: subtree}) 的决策树。
        若 tree[1] 为空则 tree[0] 为分类标记。
        若 tree[1] 不为空则 tree[0] 为最优划分属性、tree[1] 为一个字典。
        tree[1] 的 key 为最优划分属性对应的取值、value 为该取值下的分支子树。
    """
    # 若 D 中样本全为一类 C，
    # 则将 node 标记为 C 类叶结点
    label_space = get_label_space(Y=Y)
    if len(label_space) == 1:
        return label_space[0], dict()

    # 若 A 为空集，或 X 在 A 上取值相同
    # 则将 node 标记叶结点，
    # 其类别为 Y 中数量最多的类
    sample_space = get_sample_space(X=X)
    all_feature_same = True
    for feature in sample_space.keys():
        if feature not in disabled_features and len(sample_space[feature]) > 1:
            all_feature_same = False
            break
    if all_feature_same:
        return get_most_label(Y=Y), dict()

    # 从 A 中选择最优划分属性（此处基于基尼系数进行划分），
    # 将该属性记为 a
    gain = gini_index(X=X, Y=Y)
    partition = get_partition_feature(gain=gain, disabled_features=disabled_features)
    disabled_features += [partition]
    subnodes = dict()

    # 对于 a 的每一个属性值 v，
    # 为 node 生成一个分支
    # 令 D_v 表示 D 中在 a 上取值为 v 的子集
    for feature_value in sample_space[partition]:
        x, y = [], []
        m = len(X)
        for i in range(m):
            if X[i][partition] == feature_value:
                x.append(X[i])
                y.append(Y[i])
        # 若 D_v 为空集，
        # 则将分支结点标记为叶结点，
        # 其类别为 Y 中数量最多的类
        if y == []:
            subnodes[feature_value] = get_most_label(Y=Y)
        # 在 D_v 上根据 A - a 生成子决策树，连接到该分支上
        else:
            subnodes[feature_value] = build_tree(X=x, Y=y, disabled_features=disabled_features)
    return partition, subnodes


def pretty_show(tree, headers = lambda x: x, level = 0):
    """
    打印分类决策树。
    """
    if tree[1] == dict():
        print(level * '\t' + 'label: ' + tree[0])
    else:
        print(level * '\t' + 'partition feature: {}'.format(headers(tree[0])))
        for feature_value in tree[1].keys():
            print(level * '\t' + '{}'.format(feature_value))
            pretty_show(tree=tree[1][feature_value], headers=headers, level=level+1)


def predict(tree, X):
    """
    根据输入数据进行预测。
    input:
        tree: 决策树。
        X: 测试样本集合。
    output:
        对测试样本的预测值。
    """
    def __predict(tree, X):
        if tree[1] == dict():
            return tree[0]
        else:
            return __predict(tree=tree[1][X[tree[0]]], X=X)

    m = len(X)
    return [__predict(tree=tree, X=X[k]) for k in range(m)]


def test(prediction, Y):
    """
    根据对测试数据预测值和测试数据的真实值计算预测正确率。
    input:
        prediction: 预测值。
        Y: 真实值。
    """
    m = len(Y)
    right = 0
    error = 0
    for k in range(m):
        if prediction[k] == Y[k]:
            right += 1
        else:
            error += 1
    print("Right: {}, error: {}, right rate: {}".format(
        right, error, 1 if m == 0 else right / (right + error)))


X, Y, headers = load_data(
    file=os.path.join(__file__, '..', 'dataset', 'melon.txt'),
    sep=' ', has_headers=True)
tree = build_tree(X=X, Y=Y)
pretty_show(tree, headers=lambda x: headers[x])
prediction = predict(tree=tree, X=X)
test(prediction=prediction, Y=Y)
