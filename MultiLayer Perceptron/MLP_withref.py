from math import exp
from random import seed
from random import random
from sklearn.datasets import load_iris
import numpy as np
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Test training backprop algorithm
seed(1)
iris = load_iris()
train_data = iris.data[:10, :]
train_target = iris.target[:10]
train_data = np.concatenate((train_data, iris.data[50:60, :]), axis=0)
train_target = np.concatenate((train_target, iris.target[50:60]), axis=0)
n_inputs = len(train_data[0])
n_outputs = len(set([row for row in train_target]))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, [np.concatenate((x,y), axis=0) for x,y in zip(train_data, train_target)], 0.5, 20, n_outputs)
for layer in network:
	print(layer)

"""
from random import seed
from random import random
from math import exp 
#from sklearn.datasets import load_iris
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weight':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weight':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network
 
#Activation
def activate(weights, x):
    actv = weights[-1]
    for i in range(len(weights)-1):
        actv += weights[i] * x[i]
    return actv

#Transfer
def transfer(x):
    return 1.0/(1.0 + exp(-x))

def transfer_derivative(x):
    y = transfer(x)
    return y * (1-y)

#Propagation
def forward_propagate(network, rows):
    inputs = rows
    for layer in network:
        new_inputs = []
        for neuron in layer:
            actv = activate(neuron['weight'], inputs)
            neuron['outputs'] = transfer(actv)
            new_inputs.append(neuron['outputs'])
        inputs = new_inputs
    return inputs

#Backpropagation
def back_propagate(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if(i != len(network)-1):
            for j in range(len(layer)):
                error = 0
                for neuron in network[i+1]:
                    error += (neuron['weight'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['outputs'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j]*transfer_derivative(neuron['outputs'])
            
def update_weights(network, eta, rows):
    for i in range(len(network)):
        inputs = rows[-1]
        if(i != 0):
            inputs = [neuron['outputs'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(neuron)):
                neuron['weight'][j] += eta * neuron['delta'] * inputs[j]
            neuron['weight'][-1] += eta * neuron['delta']
            
def train_network(network, eta, train, n_epochs, n_outputs):
    for epoch in range(n_epochs):
        err = 0
        for rows in train:
            output = forward_propagate(network, rows)
            expected = [0 for i in range(n_outputs)] 
            expected[rows[-1]] = 1
            err += sum([(expected[i] - output[i])**2 for i in range(len(expected))])
            back_propagate(network, expected)
            update_weights(network, eta, rows)
        print("Epoch: {} Error: {}".format(epoch, err))
    
seed(1)
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 3, n_outputs)
train_network(network, 0.1, dataset, 1000, n_outputs)
for layer in network:
	print(layer)
"""