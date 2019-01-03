import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt
import neupy as ne
import pickle

class neuralNetwork:

    #TODO : add one more layer + dropout 
    def __init__(self, inputnodes, hiddennodes, hiddennodes2, outputnodes, learningrate, epochs):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.hnodes2 = hiddennodes2
        self.onodes = outputnodes
        self.lr = learningrate
        self.epochs = epochs
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.whh2 = np.random.normal(0.0, pow(self.hnodes2, -0.5), (self.hnodes2, self.hnodes))
        self.wh2o = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes2))  
        self.activation_function = lambda x: ss.expit(x)      
        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        hidden2_inputs = np.dot(self.whh2, hidden_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        final_inputs = np.dot(self.wh2o, hidden2_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        # output_errors = self.cross_entropy(final_outputs, targets)
        hidden2_errors = np.dot(self.wh2o.T, output_errors)
        hidden_errors = np.dot(self.whh2.T,hidden2_errors)
        # input_errors = np.dot(self.wih,hidden_errors)

        self.wh2o += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden2_outputs))
        self.whh2 += self.lr * np.dot((hidden2_errors * hidden2_outputs * (1.0 - hidden2_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass

    def query(self, input_list):
        inputs = np.array(input_list, ndmin = 2).T
        
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        hidden2_inputs = np.dot(self.whh2,hidden_outputs)
        hidden2_outputs = self.activation_function(hidden2_inputs)

        final_inputs = np.dot(self.wh2o, hidden2_outputs)
        final_outputs = self.activation_function(final_inputs)
        # final_outputs = self.softmax(final_inputs)
        
        return final_outputs

    def deserialize(self,path):
        f = open(path, 'rb')
        weights = pickle.load(f)
        self.wih = weights[0]
        self.whh2 = weights[1]
        self.wh2o = weights[2]
        f.close()

    def train_mnist(self,path):
        file = load(path)
        for e in range(self.epochs):
                for record in file:
                    all_values = record.split(',')
                    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                    targets = np.zeros(self.onodes) + 0.01
                    targets[int(all_values[0])] = 0.99
                    self.train(inputs,targets)      

    def test_mnist(self, path):
        file2 = load(path)
        for record in file2:
            all_values = record.split(',')
            correct_label = int(all_values[0])
            # print(correct_label, "correct answer")
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            outputs = n.query(inputs)
            label = np.argmax(outputs)
            # print(label, "network's answer")
            if( label == correct_label):
                scorecard.append(1)
            else:
                scorecard.append(0)

        scorecard_array = np.asarray(scorecard)
        print ("performance = ", scorecard_array.sum() /  scorecard_array.size)

    def softmax(self,x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def cross_entropy(self,predictions, targets, epsilon=1e-12):
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(targets*np.log(predictions+1e-9))/N
        return ce



def load(path):
    data_file = open(path, "r")
    data_list = data_file.readlines()
    data_file.close()
    return data_list

if __name__ == "__main__":
    # file = load("bigger/mnist_train.csv")
    # file2 = load("bigger/mnist_test.csv")


    input_nodes = 784
    hidden_nodes = 150
    hidden_nodes2 = 150
    output_nodes = 10
    learning_rate = 0.1
    scorecard = []
    epochs = 1

    n = neuralNetwork(input_nodes,hidden_nodes,hidden_nodes2,output_nodes,learning_rate,epochs)

    # train
    # for e in range(epochs):
    #     for record in file:
    #         all_values = record.split(',')
    #         inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #         targets = np.zeros(output_nodes) + 0.01
    #         targets[int(all_values[0])] = 0.99
    #         n.train(inputs,targets)      

    n.train_mnist("bigger/mnist_train.csv")

    n.serialize()  

    # n.deserialize()

    n.test_mnist("bigger/mnist_test.csv")

    # print(file[0].split(',')[0])
    # print(n.query(np.asfarray(file[0].split(',')[1:])))

    #query
    # for record in file2:
    #     all_values = record.split(',')
    #     correct_label = int(all_values[0])
    #     print(correct_label, "correct answer")
    #     inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #     outputs = n.query(inputs)
    #     label = np.argmax(outputs)
    #     # print(label, "network's answer")
    #     if( label == correct_label):
    #         scorecard.append(1)
    #     else:
    #         scorecard.append(0)

    # scorecard_array = np.asarray(scorecard)
    # print ("performance = ", scorecard_array.sum() /  scorecard_array.size)