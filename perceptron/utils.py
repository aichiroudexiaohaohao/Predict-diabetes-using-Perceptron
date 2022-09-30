import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data(data_path):
    """
    This function used to load txt data
    """
    X = []
    y = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split(' ')
            label = int(line[0])
            features = [0.0] * 8
            for feat in line[1:]:
                if feat == '':
                    continue
                idx, value = feat.split(':')
                idx = int(idx) - 1
                value = float(value)
                features[idx] = value
            X.append(features)
            y.append(label)
    X = np.array(X).reshape(len(X), len(X[0]))
    y = np.array(y)
    return X, y


def make_data(X, y):
    """
    This function divide data into train, test
    """
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0, test_size=0.2)
    return (train_X, train_y), (test_X, test_y)


def plot_confusion_matrix(cf_matrix, title=None):
    """
    This function used to plot confusion matrix of our result
    """
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('predicted Values')
    ax.set_ylabel('Actual Values ')

    plt.show()
