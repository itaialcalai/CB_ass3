import random
import numpy as np
from sklearn.model_selection import train_test_split

# ISSUE: The algorithm is not converging to a solution. The fitness is not decreasing.
# options: 1. change the fitness function
#          2. change the selection function
#          3. add backpropagation
#          3. change the network architecture

# A genetic algorithm to select the best neural network architecture to classify 16 bit binary sequences to 0 or 1

# class GeneticAlgorithm:
class GeneticAlgorithm:
    @staticmethod
    # execute function runs the genetic algorithm
    def execute(pop_size, generations, threshold, X, y, network):
        for gen in range(generations):
            agents = GeneticAlgorithm.generate_agents(pop_size, network)
            agents = GeneticAlgorithm.fitness(agents, X, y)
            agents = GeneticAlgorithm.selection(agents)
            print("Generation ", str(gen), " Best fitness: ", str(agents[0].fitness))

            if agents[0].fitness <= threshold:
                print("Threshold met at generation ", str(gen))
                break

            agents = GeneticAlgorithm.crossover(agents, pop_size, network)
            agents = GeneticAlgorithm.mutation(agents)
            agents = GeneticAlgorithm.fitness(agents, X, y)
        # return best network
        return agents[0].network

    # generate_agents function generates a list of agents
    @staticmethod
    def generate_agents(pop_size, network):
        return [Agent(network) for _ in range(pop_size)]

    # fitness function calculates the fitness of each agent
    @staticmethod
    def fitness(agents, X, y):
        print("Fitness")
        counter = 0
        for agent in agents:
            y_hat = agent.network.forward(X) # forward pass
            agent.fitness = np.sum(np.abs(y_hat - y)) # calculate fitness as the sum of absolute differences
            # print("Fitness for Agent ", str(counter), " is ", str(agent.fitness))
            counter += 1
        return agents

    # selection function selects the top agents by fitness
    @staticmethod
    def selection(agents):
        print("Selection")
        agents = sorted(agents, key=lambda agent: agent.fitness, reverse=False)
        # print('\n'.join(map(str, agents)))

        # pick the top 40% of agents
        agents = agents[:int(0.4 * len(agents))]
        # print("Agents selected: ", str(len(agents)))
        return agents

    # unflatten_weights function converts the flattened weights back to the original shape
    @staticmethod
    def unflatten_weights(flattened_weights, network):
        new_weights = []
        index = 0
        for layer in network:
            size = np.prod(layer)
            new_weights.append(flattened_weights[index: index + size].reshape(layer))
            index += size
        return new_weights

    # crossover function performs crossover between two agents
    @staticmethod
    def crossover(agents, pop_size, network):
        print("Crossover")
        offspring = []
        for _ in range((pop_size - len(agents)) // 2):
            # randomly select two parents
            parent1 = random.choice(agents)
            parent2 = random.choice(agents)
            # create two children
            child1 = Agent(network)
            child2 = Agent(network)
            # randomly select a split point
            shapes = [weight.shape for weight in parent1.network.weights]
            # perform crossover
            genes1 = np.concatenate([weight.flatten() for weight in parent1.network.weights])
            genes2 = np.concatenate([weight.flatten() for weight in parent2.network.weights])
            split = random.randint(0, genes1.size)
            child1_genes = np.concatenate([genes1[:split], genes2[split:]])
            child2_genes = np.concatenate([genes2[:split], genes1[split:]])
            # convert the flattened weights to the original shape
            child1.network.weights = GeneticAlgorithm.unflatten_weights(child1_genes, shapes)
            child2.network.weights = GeneticAlgorithm.unflatten_weights(child2_genes, shapes)
            # add the children to the offspring list
            offspring.append(child1)
            offspring.append(child2)
        # add the offspring to the agents list
        agents.extend(offspring)
        return agents

    # mutation function performs mutation on the agents by rate of 10%
    @staticmethod
    def mutation(agents):
        print("Mutation")
        for agent in agents:
            if random.uniform(0, 1) <= 0.1:
                # flatten the weights
                weights = agent.network.weights
                shapes = [weight.shape for weight in weights]
                flattened_weights = np.concatenate([weight.flatten() for weight in weights])
                # randomly select a gene to mutate
                mutation_index = random.randint(0, flattened_weights.size - 1)
                # assign a random value to the selected gene
                flattened_weights[mutation_index] = np.random.randn()
                new_arr = []
                index_weight = 0
                # convert the flattened weights to the original shape
                for shape in shapes:
                    size = np.prod(shape)
                    new_arr.append(flattened_weights[index_weight: index_weight + size].reshape(shape))
                    index_weight += size
                # assign the new weights to the agent
                agent.network.weights = new_arr
        return agents

# class Agent:
class Agent:
    def __init__(self, network):
        self.network = NeuralNetwork(network)
        self.fitness = 0

# NeuralNetwork class
class NeuralNetwork:
    # initialize the network
    def __init__(self, network):
        self.weights = []
        self.activations = []
        # initialize the weights and activations
        for layer in network:
            # first layer
            if layer[0] is None:
                input_size = network[network.index(layer) - 1][1]
            else:
                input_size = layer[0]
            output_size = layer[1]
            activation = layer[2]
            self.weights.append(np.random.randn(input_size, output_size))
            self.activations.append(activation)

    # forward function performs forward pass
    def forward(self, data):
        input_data = data
        for i in range(len(self.weights)):
            # apply the activation function on the dot product of input data and weights
            input_data = self.activations[i](np.dot(input_data, self.weights[i]))
        return input_data

# read_file function reads the data from the file and returns the data vectors and labels
def read_file(file_path):
    X = []
    y = []

    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            vector = list(map(int, data[0]))
            label = int(data[1])
            X.append(vector)
            y.append(label)

    return np.array(X), np.array(y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compare(y, y_hat):
    return np.sum(np.abs(y - y_hat))

# split_data function splits the data into training and testing sets based on the test_ratio
def split_data(X, y, test_ratio=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def main():
    print("Hello World!")
    # read the data from the file
    X, y = read_file('nn0.txt')
    X_train, X_test, y_train, y_test = split_data(X, y)
    # define the network
    network = [
        [16, 8, sigmoid],
        [None, 1, sigmoid]
    ]
    # define the parameters
    pop_size = 10
    generations = 1000
    threshold = 0.01
    # execute the genetic algorithm
    best_network = GeneticAlgorithm.execute(pop_size, generations, threshold, X_train, y_train.T, network)
    print("Best Network: ")
    print(best_network.weights)
    # yhat = best_network.forward(X_test)
    # print("Test Data Results: MSE: ")
    # print(compare(y_test, yhat))


if __name__ == "__main__":
    main()
