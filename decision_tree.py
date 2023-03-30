from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt

one_hot = {
    "dark-green": 0, "pitch-dark": 1, "white": 2,
    "roll-up": 0, "slighly-curled": 1, "stiff": 2,
    "dull": 0, "dead": 1, "crisp": 2,
    "clear": 0, "indistinct": 1, "blurred": 2,
    "hollow": 0, "plain": 1, "slightly-hollow": 2,
    "hard": 0, "soft": 1,
}

with open("data.txt") as f:
    data = f.readlines()[1:]

samples = []
labels = []
for _ in data:
    d = _.split(",")
    str_sample = d[1:-1]
    label = d[-1][:-1]
    sample = []
    for feature in str_sample[:-2]:
        sample.append(one_hot[feature])
    for feature in str_sample[-2:]:
        sample.append(float(feature))
    samples.append(sample)
    labels.append(label)


model = DecisionTreeClassifier(criterion="entropy")
model.fit(samples, labels)
print(model.predict(samples))

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
plot_tree(model)
fig.savefig('decision-tree.png')
