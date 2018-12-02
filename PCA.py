import csv
import random
import math
import operator
import time
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as pyplot


start_time = time.time()


def read_dataset(filename, split, training=[], test=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(len(dataset[0])):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                training.append(dataset[x])
            else:
                test.append(dataset[x])


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((float(instance1[x]) - float(instance2[x])), 2)
    return math.sqrt(distance)


def get_neighbors(training, ins, k):
    distances = []
    length = len(ins) - 1
    for x in range(len(training)):
        dist = euclidean_distance(ins, training[x], length)
        distances.append((training[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def get_response(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(iter(list(classVotes.items())), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def accuracy(test, predictions):
    correct = 0
    testapp=[]

    for x in range(len(test)):
        testapp.append(test[x][-1])
        if test[x][-1] == predictions[x]:
            correct += 1
    precision, recall, fscore, support = score(testapp, predictions, average='micro')
    accur = correct / float(len(test))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('accuracy: {}'.format(accur))
    accuracy_list.append(accur)
    precision_list.append(precision)
    recall_list.append(precision)


def eigenval_and_eigenvec(dataset):
    dataset = list(map(list, list(zip(*dataset))))
    convData = np.cov(dataset)  # Covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(convData)
    sorted_eigenvalues = np.argsort(eigenvalues)
    x = eigenvalues.shape
    feature_size = x[0]
    return sorted_eigenvalues, feature_size


def pick_features(dataset, sorted_eigenvalues, size, number_of_feature):
    new_dataset = np.zeros((int(size), number_of_feature + 1))
    for i in range(0, number_of_feature):
        for j in range(len(sorted_eigenvalues)):
            if sorted_eigenvalues[j] == i:
                for x in range(0, int(size)):
                    new_dataset[x][i] = dataset[x][j]
    for x in range(0, int(size)):
        new_dataset[x][i + 1] = dataset[x][57]
    return new_dataset


def prediction(training, test):
    predictions.clear()
    for x in range(len(test)):
        neighbors = get_neighbors(training, test[x], k)
        result = get_response(neighbors)
        predictions.append(result)

    accuracy(test, predictions)



def try_for_all_feature_subset(training, test, feature_size, sorted_eigenvalues):
    for i in range(1, feature_size):
        new_dataset = pick_features(training, sorted_eigenvalues, m, i)
        new_test = pick_features(test, sorted_eigenvalues, n, i)
        print(new_dataset.shape, " ", new_test.shape)
        new_dataset = new_dataset.tolist()
        new_test = new_test.tolist()
        prediction(new_dataset, new_test)

accuracy_list = []
precision_list = []
recall_list = []
training = []
test = []
predictions = []
split = 0.50
k = 5
read_dataset('dataset.txt', split, training, test)
m = repr(len(training))
n = repr(len(test))
print('Train set: ' + m)
print('Test set: ' + n)

prediction(training, test)

# accuracy, precision, and recall

sorted_eigenvalues, feature_size = eigenval_and_eigenvec(training)
print("PCA Phase")
try_for_all_feature_subset(training, test, feature_size, sorted_eigenvalues)

pyplot.xlabel('Iterations')
pyplot.ylabel('Accuracy')
pyplot.title('Accuracy Plot')
pyplot.plot(list(range(len(accuracy_list))), accuracy_list)
pyplot.show()

pyplot.xlabel('Iterations')
pyplot.ylabel('Precision')
pyplot.title('Precision Plot')
pyplot.plot(list(range(len(precision_list))), precision_list)
pyplot.show()

pyplot.xlabel('Iterations')
pyplot.ylabel('Recall')
pyplot.title('Recall Plot')
pyplot.plot(list(range(len(recall_list))), recall_list)
pyplot.show()


print("--- %s seconds ---" % (time.time() - start_time))
