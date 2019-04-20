import matplotlib.pyplot as plt

from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp

# Opening dataset file
def read_csv(filename):
	iris_dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			iris_dataset.append(row)
	return iris_dataset[1:151]

iris_dataset = read_csv('iris_random.csv')

# Convert string column to float
for i in range(len(iris_dataset[0])-1):
	for row in iris_dataset:
		row[i] = float(row[i].strip())

# String column to integet
column = (len(iris_dataset[0])-1)
class_values = [row[column] for row in iris_dataset]
unique = set(class_values)
lookup = dict()
for i, value in enumerate(unique):
	lookup[value] = i
for row in iris_dataset:
	row[column] = lookup[row[column]]

print(iris_dataset)

# Network Initialization
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
 
# Feed forward input ke network output
def feed_forward(network, row):
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
 
# Update error to previous layer's neurons
def backprop_update_error(network, expected):
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
def update_weight(network, row, learning_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += learning_rate * neuron['delta']

def predict(network, row):
	outputs = feed_forward(network, row)
	return outputs.index(max(outputs))

# Train a network for a fixed number of epochs
def train_network(network, train, learning_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		accuracy = 0
		for row in train:
			outputs = feed_forward(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backprop_update_error(network, expected)
			update_weight(network, row, learning_rate)
			prediction = predict(network, row)
			if (row[-1] == prediction):
				accuracy += 1
		epoch_accuracy_train.append(accuracy/len(train))
		epoch_error_train.append(sum_error/len(train))
		print('Training: ')
		print('>epoch=%d, lrate=%.3f, error=%.3f, accuracy =%.3f' % (epoch+1, learning_rate, sum_error/len(train), accuracy/len(train)))
    
def validate(network, validate_set, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		accuracy = 0
		for row in validate_set:
			outputs = feed_forward(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			prediction = predict(network, row)
			if (row[-1] == prediction):
				accuracy += 1
		epoch_accuracy_validate.append(accuracy/len(validate_set))
		epoch_error_validate.append(sum_error/len(validate_set))


seed(1)
epoch_error_train = list()
epoch_accuracy_train = list()
epoch_error_validate = list()
epoch_accuracy_validate = list()
train_set = iris_dataset[:120]
validate_set = iris_dataset[120:]

#Train
learning_rate = 0.1
n_epoch = 100
n_inputs = len(iris_dataset[0]) - 1
n_outputs = len(set([row[-1] for row in iris_dataset]))
print(n_outputs)
network = initialize_network(n_inputs, 3, n_outputs)
train_network(network, train_set, learning_rate, n_epoch, n_outputs)

#Validate
validate(network, validate_set, n_epoch, n_outputs)
print('validate: ')
print(epoch_error_validate)
print(epoch_accuracy_validate)
print('>Error=%.3f, accuracy =%.3f' % (epoch_error_validate[-1], epoch_accuracy_validate[-1]))

plt.figure(1)
plt.plot(epoch_error_train, label = 'Training')
plt.plot(epoch_error_validate, label = 'validate')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(epoch_accuracy_train, label = 'Training')
plt.plot(epoch_accuracy_validate, label = 'validate')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()