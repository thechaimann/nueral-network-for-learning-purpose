import random

class Neuron:
    def __init__(self, Size, randomize=True, Leak=0.01):
        if randomize:
            self.bias = random.uniform(-1, 1)
            self.weights = [random.uniform(-1, 1) for _ in range(Size)]
        else:
            self.bias = 0
            self.weights = [0] * Size
        
        self.leak = Leak
        self.last_input = []
        self.last_output_before_activation = 0
        self.gradient = 0

    def activate(self, x):
        if x > 0:
            return x
        else:
            return self.leak * x

    def forward(self, inputs):
        self.last_input = inputs
        self.last_output_before_activation = self.bias + sum(w * i for w, i in zip(self.weights, inputs))
        return self.activate(self.last_output_before_activation)

class Layer:
    def __init__(self, num_neurons, input_size, Leak=0.01):
        self.neurons = [Neuron(input_size, Leak=Leak) for _ in range(num_neurons)]

    def forward(self, inputs):
        return [n.forward(inputs) for n in self.neurons]

class Network:
    def __init__(self, layer_sizes, input_size, Leak=0.01):
        self.layers = []
        current_input_size = input_size
        for size in layer_sizes:
            self.layers.append(Layer(size, current_input_size, Leak=Leak))
            current_input_size = size

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def train(self, inputs, targets, learning_rate):
        output = self.forward(inputs)

        # Output Layer
        output_layer = self.layers[-1]
        for i in range(len(output_layer.neurons)):
            neuron = output_layer.neurons[i]
            error = targets[i] - output[i]
            derivative = 1 if neuron.last_output_before_activation > 0 else neuron.leak
            neuron.gradient = error * derivative

        # Hidden Layers
        for l in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[l]
            next_layer = self.layers[l+1]
            for i in range(len(current_layer.neurons)):
                neuron = current_layer.neurons[i]
                contribution_to_error = sum(next_n.weights[i] * next_n.gradient for next_n in next_layer.neurons)
                derivative = 1 if neuron.last_output_before_activation > 0 else neuron.leak
                neuron.gradient = contribution_to_error * derivative

        # Update weights and biases
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.bias += learning_rate * neuron.gradient
                for i in range(len(neuron.weights)):
                    neuron.weights[i] += learning_rate * neuron.gradient * neuron.last_input[i]

class Trainer:
    def __init__(self, network):
        self.network = network

    def train(self, training_data, iterations, learning_rate):
        # Using a variable to decide when to print progress
        debugInterval = iterations // 10 
        
        for i in range(iterations):
            total_loss = 0
            for inputs, targets in training_data:
                # Calculate loss before training step for progress tracking
                output = self.network.forward(inputs)
                total_loss += sum((t - o) ** 2 for t, o in zip(targets, output)) / len(targets)
                
                self.network.train(inputs, targets, learning_rate)
            
            # Print progress
            if i % debugInterval == 0:
                avg_loss = total_loss / len(training_data)
                print(f"Iteration {i:5} | Average Loss: {avg_loss:.6f}")

if __name__ == "__main__":
    #example usage
    Input_Size = 2
    hidden_layer = [4, 1]
    LearnRate = 0.05
    i = 5000
    leak_value = 0.01

    net = Network(layer_sizes=hidden_layer, input_size=Input_Size, Leak=leak_value)
    teacher = Trainer(net)

    train_data = [
        ([0, 0], [0]), 
        ([0, 1], [1]), 
        ([1, 0], [1]), 
        ([1, 1], [0])
    ]

    print("Training...")
    teacher.train(train_data, i, LearnRate)
    print("Done.\n")

    for x, y in train_data:
        res = net.forward(x)
        print(f"In: {x} | Target: {y} | Out: {round(res[0], 4)}")
    # Use these as probablities wher you select the highest one as the output