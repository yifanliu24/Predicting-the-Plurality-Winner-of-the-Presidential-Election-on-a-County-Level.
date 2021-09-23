
import pandas as pd
import numpy as np
import scipy.special
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from str2bool import str2bool
import sys

hyper_flag = str2bool(sys.argv[1])
hidden_1 = int(sys.argv[2])
hidden_2 = int(sys.argv[3])
lr_start = float(sys.argv[4])
lr_stop = float(sys.argv[5])

data = "./county_statistics.csv"

pandas_test = pd.read_csv(data, index_col=0)

cases_100k_np = pandas_test['cases_100k'].to_numpy().reshape((-1, 1))
male_frac_np = pandas_test['Male_frac'].to_numpy().reshape((-1, 1))
non_white_np = pandas_test['Non-white'].to_numpy().reshape((-1, 1))
income_cap_np = pandas_test['IncomePerCap'].to_numpy().reshape((-1, 1))
college_np = pandas_test['college_completion'].to_numpy().reshape((-1, 1))
trump_np = pandas_test['Trump_win'].to_numpy().reshape((-1, 1))
total_np = np.concatenate([cases_100k_np, male_frac_np, non_white_np,
                           income_cap_np, college_np, trump_np], axis=1)


def sample_no_replacement(data_np, k):
    size = data_np.shape[0]
    index_list = list(range(size))
    fold_size = size//k
    final_list = [None] * k
    for i in range(k):
        if i < (k-1):
            # make each entry in here a line in total
            # pop sampled numbers
            this_fold_numbers = random.sample(index_list, fold_size)
            index_list = [ele for ele in index_list if ele not in this_fold_numbers]
            final_list[i] = [data_np[j] for j in this_fold_numbers]
        else:
            # fill last entry with remaining
            final_list[i] = [data_np[j] for j in index_list]
    return final_list


# neural network class definition
class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes1, hiddennodes2, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.h1nodes = hiddennodes1
        self.h2nodes = hiddennodes2
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih1 = np.random.normal(0.0, pow(self.inodes, -0.5), (self.h1nodes, self.inodes))
        self.wh1h2 = np.random.normal(0.0, pow(self.h1nodes, -0.5), (self.h2nodes, self.h1nodes))
        self.wh2o = np.random.normal(0.0, pow(self.h2nodes, -0.5), (self.onodes, self.h2nodes))

        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)



    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # print(inputs)
        # print(targets)
        # calculate signals into hidden layer
        hidden1_inputs = np.dot(self.wih1, inputs)
        # calculate the signals emerging from hidden layer 1
        hidden1_outputs = self.activation_function(hidden1_inputs)

        hidden2_inputs = np.dot(self.wh1h2, hidden1_outputs)
        # calculate the signals emerging from hidden layer 2
        hidden2_outputs = self.activation_function(hidden2_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.wh2o, hidden2_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden2_errors = np.dot(self.wh2o.T, output_errors)
        hidden1_errors = np.dot(self.wh1h2.T, hidden2_errors)

        # update the weights for the links between the hidden2 and output layers
        self.wh2o += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        np.transpose(hidden2_outputs))

        # update the weights for the links between the hidden1 and hidden2
        self.wh1h2 += self.lr * np.dot((hidden2_errors * hidden2_outputs * (1.0 - hidden2_outputs)),
                                        np.transpose(hidden1_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih1 += self.lr * np.dot((hidden1_errors * hidden1_outputs * (1.0 - hidden1_outputs)),
                                        np.transpose(inputs))



    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden1_inputs = np.dot(self.wih1, inputs)
        # calculate the signals emerging from hidden layer
        hidden1_outputs = self.activation_function(hidden1_inputs)

        hidden2_inputs = np.dot(self.wh1h2, hidden1_outputs)
        # calculate the signals emerging from hidden layer
        hidden2_outputs = self.activation_function(hidden2_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.wh2o, hidden2_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# number of input, hidden and output nodes
input_nodes = 5
output_nodes = 1
K = 10
def find_best_arch(h1, h2, lr_list):
    best_score = 0
    best_h1 = 0
    best_h2 = 0
    accuracies = []
    print("Hidden 1 Nodes", "Hidden 2 Nodes", "Learning rate", "Accuracy", sep="\t")
    for a in range(h1+1):
        for b in range(h2+1):
            for c in lr_list:
                hidden1_nodes = a+1
                hidden2_nodes = b+1
                learning_rate = c
                NN_list = []
                for i in range(K):
                    n = neuralNetwork(input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate)
                    NN_list.append(n)
                data_list = sample_no_replacement(total_np, 10)
                my_scores = []
                for i in range(K):  # different iterations
                    training_data_list = [data_list[z] for z in range(K) if z != i]
                    test_list = [data_list[z] for z in range(K) if z == i]


                    for j in range(len(training_data_list)):
                        for m in range(len(training_data_list[j])):
                            these_inputs = np.array(training_data_list[j][m][0:-1])
                            # print("inputs",these_inputs)
                            these_targets = np.array(training_data_list[j][m][-1])
                            # print("target",these_targets)
                            NN_list[i].train(these_inputs, these_targets)

                    these_scores = []
                    for j in range(len(test_list)):
                        for m in range(len(test_list[j])):
                            test_inputs = np.array(test_list[j][m][0:-1])
                            this_target = int(round(test_list[j][m][-1], 0))
                            label = int(round(NN_list[i].query(test_inputs)[0][0], 0))
                            if label == this_target:
                                these_scores.append(1)
                    my_scores.append(sum(these_scores)/len(test_list[0]))
                score = sum(my_scores)/len(my_scores)
                if score > best_score:
                    best_score = score
                    best_h1 = hidden1_nodes
                    best_h2 = hidden2_nodes
                    best_lr = c
                print(hidden1_nodes, hidden2_nodes, c, score, sep="\t")
                accuracies.append([hidden1_nodes, hidden2_nodes, score, c])
    return best_h1, best_h2, best_score, accuracies, best_lr



def train_epochs(epochs, nn_list):
    #train_nn on number of epochs
    best_epochs = 1
    best_score = 0
    data_np = np.concatenate([cases_100k_np, male_frac_np, non_white_np,
                           income_cap_np, college_np], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data_np, trump_np, test_size=0.33)
    accuracy_over_time = []
    best_nn = neuralNetwork(nn_list[0], nn_list[1], nn_list[2], nn_list[3], nn_list[4])
    for j in range(epochs):
        for i in range(X_train.shape[0]):
            best_nn.train(X_train[i], y_train[i])
        scores = []
        for k in range(X_test.shape[0]):
            label = int(round(best_nn.query(X_test[k])[0][0], 0))
            target = int(round(y_test[k][0]))
            if label == target:
                scores.append(1)
        final_score = sum(scores)/(y_test.shape[0])
        # print(j, final_score, sum(scores), y_test.shape[0])
        if final_score > best_score:
            best_epochs = j+1
            best_score = final_score
        accuracy_over_time.append([j, final_score])
    return best_epochs, best_score, accuracy_over_time

if hyper_flag:
    desired_h1 = hidden_1
    desired_h2 = hidden_2
    lrs = np.linspace(lr_start, lr_stop, 20)
    optim_h1, optim_h2, optim_score, output_accuracies, output_lrs = find_best_arch(desired_h1, desired_h2, lrs)
    print("The best network has: ", "\n",
          optim_h1, " nodes in hidden layer 1", "\n",
          optim_h2, " nodes in hidden layer 2", "\n",
          output_lrs, "best LR",
          "and an accuracy of ", optim_score)
    prime_epoch, prime_score, accuracies_to_plot = train_epochs(100, [5, optim_h1, optim_h2, 1, output_lrs])
    print("The ideal system is trained using", prime_epoch, "epochs", "\n",
          "and has an accuracy of", prime_score)
    x_plotting = [item[0] for item in accuracies_to_plot]
    # print(x_plotting)
    y_plotting = [item[1] for item in accuracies_to_plot]
    # print(y_plotting)
    plt.plot(x_plotting, y_plotting)
    plt.xlabel('Number of epochs')
    plt.ylabel('Performance')
    plt.title('Performance of the best neural network architecture over training epochs')
    plt.show()
else: #already know best struct
    prime_epoch, prime_score, accuracies_to_plot = train_epochs(100, [5, hidden_1, hidden_2, 1, lr_start])
    print("The ideal system is trained using", prime_epoch, "epochs", "\n",
          "and has an accuracy of", prime_score)
    x_plotting = [item[0] for item in accuracies_to_plot]
    # print(x_plotting)
    y_plotting = [item[1] for item in accuracies_to_plot]
    # print(y_plotting)
    plt.plot(x_plotting, y_plotting)
    plt.xlabel('Number of epochs')
    plt.ylabel('Performance')
    plt.title('Performance of the best neural network architecture over training epochs')
    plt.show()
