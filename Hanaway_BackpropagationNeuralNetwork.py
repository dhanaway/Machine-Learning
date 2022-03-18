#!/usr/bin/env python3
import numpy as np
from io import StringIO

NUM_FEATURES = 124  # features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = "/Users/daltonhanaway/Downloads/data/"  

class weightsModel:
    def __init__(self, inputToHidden, hiddenToOut):
        self.inputToHidden = inputToHidden
        self.hiddenToOut = hiddenToOut

    a_hiddenLayer = np.ndarray([])
    hiddenLayer = np.ndarray([])

    a_outLayer = np.ndarray([])
    outLayer = np.ndarray([])


# returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    y = max(y, 0)  # treat -1 as 0 instead, because sigmoid's range is 0-1
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature - 1] = value
    x[-1] = 1  # bias
    return y, x


# return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals], [v[1] for v in vals])
        return np.asarray([ys], dtype=np.float32).T, np.asarray(xs, dtype=np.float32).reshape(len(xs),
                                                                                              NUM_FEATURES)  # returns a tuple, first is an array of labels, second is an array of feature vectors

def init_model(args):

    w1 = None
    w2 = None

    if args.weights_files:
        with open(args.weights_files[0], 'r') as f1:
            w1 = np.loadtxt(f1)
        with open(args.weights_files[1], 'r') as f2:
            w2 = np.loadtxt(f2)
            w2 = w2.reshape(1, len(w2))
    else:
        w1 = np.random.rand(args.hidden_dim, NUM_FEATURES)  # bias included in NUM_FEATURES
        w2 = np.random.rand(1, args.hidden_dim + 1)  # add bias column

    # At this point, w1 has shape (hidden_dim, NUM_FEATURES) and w2 has shape (1, hidden_dim + 1). In both, the last column is the bias weights.
    model = weightsModel(w1, w2)

    return model


def activation(weights, inputs):
    return np.dot(weights, inputs)


def g(activ):
    return 1/(1+np.exp(-activ))


def g_prime(activ_d):
    return np.exp(-activ_d) / ((np.exp(-activ_d) + 1) ** 2)


def cost_func(ak, y_correct):

    if type(y_correct) == type(1.00):
        y_correct = np.array(y_correct)

    sum_ak = np.sum(np.exp(ak))
    log_like = 0
    for i in range(len(y_correct)):
        y_k = y_correct[i]
        # have to change ak to each point corresponding to y
        y_k_hat = np.exp(ak)/sum_ak
        log_like += y_k * y_k_hat

    return log_like

def forwward_propogation(model, inputX, y_dp):

    # hidden layer calculations for a and g
    model.a_hiddenLayer = np.dot(model.inputToHidden, inputX)
    model.hiddenLayer = np.append(g(model.a_hiddenLayer), 1)

    # a and g outlayer for this data point
    model.a_outLayer = np.dot(model.hiddenToOut, model.hiddenLayer)
    model.outLayer = g(model.a_outLayer)

    return model


def back_propogation(model, trainX, trainY, args):

    topDerv = trainY - model.outLayer

    gradient2 = np.multiply(topDerv, model.hiddenLayer.T)

    bottomDerve = topDerv * model.hiddenToOut[0][:-1] * g_prime(model.a_hiddenLayer).T

    gradient1 = []
    for items in list(range(len(bottomDerve))):
        i = bottomDerve[items]
        columns = []
        for j in trainX:
            columns.append(i * j)
        gradient1.append(columns)
    gradient1 = np.array(gradient1)


    model.hiddenToOut = model.hiddenToOut + args.lr * gradient2
    model.inputToHidden = model.inputToHidden + args.lr * gradient1

    return model


def train_model(model, train_ys, train_xs, dev_ys, dev_xs, args):

    if (args.nodev):
        theX = train_xs
        theY = train_ys
    else:
        theX = dev_xs
        theY = dev_ys

    it_on = 0
    while it_on <= args.iterations:
        for n in list(range(len(theX))):
            model = forwward_propogation(model, theX[n], theY[n])
            model = back_propogation(model, theX[n], theY[n], args)

        it_on += 1

    return model


def test_accuracy(model, test_ys, test_xs):
    countOfPoints = np.size(test_ys)
    correct = 0
    for dp in list(range(countOfPoints)):

        #forward iteration
        model.a_hiddenLayer = np.dot(model.inputToHidden, test_xs[dp])
        model.hiddenLayer = np.append(g(model.a_hiddenLayer), 1)
        model.a_outLayer = np.dot(model.hiddenToOut, model.hiddenLayer)
        model.outLayer = g(model.a_outLayer)

        #compare
        predictValue = round(model.outLayer[0])
        if predictValue == round(test_ys[dp][0]):
            correct += 1

    accuracy = correct/countOfPoints

    return accuracy


def extract_weights(model):
    w1 = model.inputToHidden
    w2 = model.hiddenToOut

    return w1, w2

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Neural network with one hidden layer, trainable with backpropagation.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=0.3, help='Learning rate to use for update in training loop.')

    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument('--weights_files', nargs=2, metavar=('W1', 'W2'), type=str,
                               help='Files to read weights from (in format produced by numpy.savetxt). First is weights from input to hidden layer, second is from hidden to output.')
    weights_group.add_argument('--hidden_dim', type=int, default=8, help='Dimension of hidden layer.')

    parser.add_argument('--print_weights', action='store_true', default=True,
                        help='If provided, print final learned weights to stdout (used in autograding)')

    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH, 'a7a.train'),
                        help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH, 'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH, 'a7a.test'), help='Test data file.')

    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.weights_files: iterable of str; if present, contains two fields, the first is the file to read the first layer's weights from, second is for the second weight matrix.
    args.hidden_dim: int; number of hidden layer units. If weights_files is provided, this argument should be ignored.
    args.train_file: str; file to load training data from.
    args.dev_file: str; file to load dev data from.
    args.test_file: str; file to load test data from.
    """

    train_ys, train_xs = parse_data(args.train_file)

    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs = parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)

    model = init_model(args)
    model = train_model(model, train_ys, train_xs, dev_ys, dev_xs, args)

    accuracy = test_accuracy(model, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))

    if args.print_weights:
        w1, w2 = extract_weights(model)
        with StringIO() as weights_string_1:
            np.savetxt(weights_string_1, w1)
            print('Hidden layer weights: {}'.format(weights_string_1.getvalue()))
        with StringIO() as weights_string_2:
            np.savetxt(weights_string_2, w2)
            print('Output layer weights: {}'.format(weights_string_2.getvalue()))


if __name__ == '__main__':
    main()
