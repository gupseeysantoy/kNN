"""
 Created on Sun Feb 16 20:01:46 2020
 Author: Gupse Eyşan Toy
 Explanation: This code implement the k-NN classification algorithm and test it on the Iris dataset.

"""

# import libraries
import operator  # in order to using short parts
from collections import \
    defaultdict  # defaultdict allows the caller to specify the default up front when the container is started.

import matplotlib.pyplot as plt  # in order to using plot our data
import numpy as np  # in order to using create numpy arrays
import pandas as pd  # in order to using data frame
from matplotlib.colors import ListedColormap

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def plot_data(x, y):
    """
    This function plot draws the appropriate graph for the data thus using the numpy arrays.
    This method including three parameters these are x,y and title.
    X is test, y is predictions and z is title.
    """
    plt.figure(figsize=(5, 5))

    # plot Setosa
    plt.scatter(x[:, :2][y == 1, 0], x[:, :2][y == 1, 1], c='#FEC8D8')
    # plot Versicolor
    plt.scatter(x[:, :2][y == 2, 0], x[:, :2][y == 2, 1], c='#B9D6F3')
    # plot Virginica
    plt.scatter(x[:, :2][y == 3, 0], x[:, :2][y == 3, 1], c="#ADE6D0")

    plt.legend(['Setosa', 'Versicolor', 'Virginica'])
    plt.xlabel('Sepal Length(cm)')
    plt.ylabel('Petal Width(cm)')
    title = "Decision boundaries " + str(k) + " neighbors were used in kNN"
    plt.title(title);


def knn(training, test, k, method):
    """
    This method including four parameters these are training, test, k, method.
    We define training and test. First 30 samples from each flower class into the training set
    and put the rest of the samples into the test set.
    We take the value of k from the user. Besides user choose method. (euclidean or manhattan)
    """

    distances_list = []
    neighbors_list = []

    # This part calculate euclidean distances for choosing k
    if method == 'euclidean':

        for x in range(len(training)):
            x_point = test[-3:2]
            y_point = training[x][-3:2]
            dist = np.linalg.norm(x_point - y_point)
            distances_list.append((training[x], dist))
        distances_list.sort(key=lambda elem: elem[1])

        for x in range(k):
            neighbors_list.append(distances_list[x][0])
        return neighbors_list

    # This part calculate manhattan distances for choosing k
    elif method == 'manhattan':

        for x in range(len(training)):
            x_point = test[-3:2]
            y_point = training[x][-3:2]
            dist = sum(abs(x_point - y_point))
            distances_list.append((training[x], dist))
        distances_list.sort(key=lambda elem: elem[1])

        for x in range(k):
            neighbors_list.append(distances_list[x][0])
        return neighbors_list

    else:
        print("Your method choose is wrong, please enter again")


if __name__ == "__main__":
    """
     This main part performs the application of knn using knn and plot_data methods and than draws the data obtained as a result. 
    """

    # initializing
    predictions = []
    sepalLength = 0
    petalWidth = 3

    # read txt file and give column names with using the pandas dataframe
    columnNames = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'label']
    data = pd.read_csv("iris_data.txt", delimiter=',', names=columnNames, header=None)

    # create dataFrame using data firt and third column: SepalLength and PetalWidth and also using fourth column this is label.
    dataFrame = data.iloc[:, [0, 3, 4]]

    # create labels for this using the last column in the dataFrame
    labels = dataFrame.iloc[:, -1]

    # divide training sets first 30 elements for each different label
    train_data1 = dataFrame.iloc[0:30]
    train_label1 = labels.iloc[0:30]
    train_data2 = dataFrame.iloc[50:80]
    train_label2 = labels.iloc[50:80]
    train_data3 = dataFrame.iloc[100:130]
    train_label3 = labels.iloc[100:130]

    # divide test sets first 30 elements for each different label
    test_data1 = dataFrame.iloc[30:50]
    test_label1 = labels.iloc[30:50]
    test_data2 = dataFrame.iloc[80:100]
    test_label2 = labels.iloc[80:100]
    test_data3 = dataFrame.iloc[130:]
    test_label3 = labels.iloc[130:]

    # append the discrete training data and create to be used training data
    training = train_data1.append(train_data2).append(train_data3).values
    trainingl = train_label1.append(train_label2).append(train_label3).values

    # append the discrete test data and create to be used test data
    test = test_data1.append(test_data2).append(test_data3).values
    testl = test_label1.append(test_label2).append(test_label3).values

    # convet training and test dataframe part
    training_df = pd.DataFrame(training)
    test_df = pd.DataFrame(test)

    # Create numpy array using data
    data = data.iloc[:, [0, 3, 4]].values

    method = input('Choose euclidean or manhattan)\n')
    print("The chosen distance method is:", method + "\n")
    k = int(input('Choose number k (k=1,3,5,7,9,11,15) \n'))

    # convert labels
    for i in range(len(data)):
        if data[i][2] == 'Iris-setosa':
            data[i][2] = 1
        elif data[i][2] == 'Iris-versicolor':
            data[i][2] = 2
        elif data[i][2] == 'Iris-virginica':
            data[i][2] = 3

    label = data[:, 2]
    title = "Original data"
    # plot_data(data, label, title)

    # In this section, the majority vote of the neighbors of the data point is taken as an estimate.
    # Finally, the lists the votes and estimates are found.
    for i in range(len(test)):

        neighbors = knn(training, test[i], k, method)
        throw_out = defaultdict(list)
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in throw_out:
                throw_out[response] += 1
            else:
                throw_out[response] = 1
        sortedVote = sorted(throw_out.items(), key=operator.itemgetter(1), reverse=True)
        sortedVoteij = 0
        result = sortedVote[sortedVoteij][sortedVoteij]
        predictions.append(result)

    # Calculate accuracy.
    correct = 0
    for x in range(len(test)):

        if predictions[x] == test[x][-1]:
            correct += 1
        else:
            correct = correct

    count = len(test) - correct
    print('\nError Count:', count, '/', len(test))
    accuracy = (correct * 100) / float(len(test))
    print('Accuracy: ' + "%.2f" % float(str(accuracy)))

    # convert labels
    for i in range(len(predictions)):
        if predictions[i] == 'Iris-setosa':
            predictions[i] = 1
        elif predictions[i] == 'Iris-versicolor':
            predictions[i] = 2
        elif predictions[i] == 'Iris-virginica':
            predictions[i] = 3

    predictions = np.asarray(predictions, dtype=np.float32)

    # plot the knn prediction results

    plot_data(test, predictions)
    plt.show()
