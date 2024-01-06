import numpy as np

# Define the sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.hidden_weight = np.random.rand(input_nodes, hidden_nodes) #hidden layer weight
        self.output_weight = np.random.rand(hidden_nodes, output_nodes) #output layer weight
        self.hidden_bias = np.random.rand(hidden_nodes) #hidden layer bias
        self.output_bias = np.random.rand(output_nodes) #output layer bias

    def feedforward(self, input):
        self.hidden_layer = sigmoid(np.dot(input, self.hidden_weight) + self.hidden_bias)
        self.output = sigmoid(np.dot(self.hidden_layer, self.output_weight) + self.output_bias)
        
    def backpropagation(self, input, expected_output):
        # Calculate error
        output_error = expected_output - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.output_weight.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer)

        # Update weights and biases
        self.output_weight += self.hidden_layer.T.dot(output_delta)
        self.output_bias += np.sum(output_delta, axis=0)
        self.hidden_weight += input.T.dot(hidden_delta)
        self.hidden_bias += np.sum(hidden_delta, axis=0)

    def train(self, input, expected_output, epochs=10000):
        for _ in range(epochs):
            self.feedforward(input)
            self.backpropagation(input, expected_output)

if "__main__" == __name__:
    inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 4, 1
    functions = ["AND", "OR", "XOR"]
    inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    expected_data = {
        "AND": np.array([[0],[0],[0],[1]]),
        "OR": np.array([[0],[1],[1],[1]]),
        "XOR": np.array([[0],[1],[1],[0]])
    }

    for function in functions:
        print(f"Training for {function}")
        neural_network = NeuralNetwork(inputLayerNeurons,hiddenLayerNeurons, outputLayerNeurons)
        neural_network.train(inputs,expected_data[function])

        # Test the trained network
        for test_input in inputs:
            neural_network.feedforward(test_input)
            print(f"{function}({test_input}) = {round(neural_network.output[0], 6)}")