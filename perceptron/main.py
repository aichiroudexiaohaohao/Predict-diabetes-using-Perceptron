from model import *
from utils import *



def exp_on_origin_data():
    """
    Run experiment On Original Data

    """
    X, y = load_data('data.txt')
    train_data, test_data = make_data(X, y)
    model = Perceptron()
    model.fit(train_data[0], train_data[1])
    predictions = model.predict(test_data[0])
    acc = accuracy_score(predictions, test_data[1])
    conf = confusion_matrix(predictions, test_data[1])
    return acc, conf


def exp_on_scaled_data():
    """
    Run experiment On Scaled Data

    """
    X, y = load_data('scaled_data.txt')
    train_data, test_data = make_data(X, y)
    model = Perceptron()
    model.fit(train_data[0], train_data[1])
    predictions = model.predict(test_data[0])
    acc = accuracy_score(predictions, test_data[1])
    conf = confusion_matrix(predictions, test_data[1])
    return acc, conf


if __name__ == '__main__':
    ori_acc, conf = exp_on_origin_data()
    print('Accuracy on origin data is %.2f' % ori_acc)
    plot_confusion_matrix(conf, 'Confusion Matrix with labels on origin data')

    sca_acc, conf = exp_on_scaled_data()
    print('Accuracy on origin data is %.2f' % sca_acc)
    plot_confusion_matrix(conf, 'Confusion Matrix on scaled data')
