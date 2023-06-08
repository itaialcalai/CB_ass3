# Itai Alcalai 2060071110
import random
import numpy as np
from sklearn.model_selection import train_test_split

# A genetic algorithm to select the best neural network architecture to classify 16 bit binary sequences to 0 or 1

# Performance: Avarage accuracy on test set is over 0.9 (10% errors)

# Issue: The algorithm improves very slowly after ~500 generations due to homogeneous population and lack of NN variation. Sometimes converges to local minima.
# options: 1. 
#          2. add backpropagation - the weights are being updated only by the genetic algorithm -> maybe not allowed ?
#          3. change hyperparameters - pop_size, generations, threshold, mutation_rate and so on
#          4. change fitness function
#          5. change the network architecture

# TODO: 1. tend to the issue
#       2. check nn1.txt results
#       2. change user file input
#       3. other assignment requerments
#       4. write report


# class GeneticAlgorithm:
class GeneticAlgorithm:
    @staticmethod
    # execute function runs the genetic algorithm
    def execute(pop_size, generations, threshold, X, y, network):
        #main loop
        agents = GeneticAlgorithm.generate_agents(pop_size, network)
        agents = GeneticAlgorithm.fitness(agents, X, y)
        beta = 1 # beta for cost calculation
        child_ratio = 2 # ratio of children to generate
        mute_rate = 0.2 # intial mutation rate
        last_fitness = None # fitness of the last generation init
        stuck = 0 # counter for stuck at local minima
        # run genetic algorithm
        for gen in range(generations):
            costs = np.array([x.fitness for x in agents])
            avg_cost = np.mean(costs)
            if avg_cost != 0:
                costs = costs / avg_cost
            probs = np.exp(-beta * costs)
            probs = probs / np.sum(probs)
            child_pop = []
            nc = int(np.round(child_ratio * pop_size / 2) * 2) # number of children to generate
            for _ in range(nc // 2):
                # select parents
                parent1 = agents[GeneticAlgorithm.roulette_wheel_selection(probs)]
                parent2 = agents[GeneticAlgorithm.roulette_wheel_selection(probs)]
                # crossover
                child1, child2 = GeneticAlgorithm.crossover(parent1, parent2, network)
                # mutate
                child1 = GeneticAlgorithm.mutation(child1, mute_rate)
                child2 = GeneticAlgorithm.mutation(child2, mute_rate)
                # check if child is better than best person
                # add to child population
                child_pop.append(child1)
                child_pop.append(child2)
            child_pop = GeneticAlgorithm.fitness(child_pop, X, y)
            # select survivors
            agents += child_pop
            agents = GeneticAlgorithm.selection(agents, pop_size)

            # check if threshold is met
            if agents[0].fitness <= threshold:
                print("Threshold met at generation ", str(gen))
                break
            # check if stuck at local minima
            if agents[0].fitness == last_fitness:
                stuck +=1
                if stuck == 100:
                    print("Stuck at local minima at generation ", str(gen))
                    break
            else:
                stuck = 0
            last_fitness = agents[0].fitness
            print("Generation ", str(gen), " Best fitness: ", str(agents[0].fitness))
        return agents[0].network

    # generate_agents function generates a list of agents
    @staticmethod
    def generate_agents(pop_size, network):
        return [Agent(network) for _ in range(pop_size)]

    # fitness function calculates the fitness of each agent
    @staticmethod
    def fitness(agents, X, y):
        # print("Fitness")
        # counter = 0
        for agent in agents:
            y_hat = agent.network.forward(X) # forward pass
            # reshape y_hat to match y
            y_hat = y_hat.reshape(y.shape)
            # calculate fitness as the sum of absolute differences
            agent.fitness = np.sum(np.abs(y_hat - y)) # calculate fitness as the sum of absolute differences
            # print("Fitness for Agent ", str(counter), " is ", str(agent.fitness))
            # counter += 1
        return agents

    @staticmethod
    # roulette wheel selection method
    def roulette_wheel_selection(fitness_values):
        cumulative_fitness = np.cumsum(fitness_values)  # Calculate the cumulative sum of fitness values
        selection_point = sum(fitness_values) * np.random.rand()  # Generate a random selection point
        index = np.argwhere(selection_point <= cumulative_fitness)[0][0]  # Find the index where selection_point falls
        return index  # Return the selected individual

    # selection function selects the top agents by fitness
    @staticmethod
    def selection(agents, pop_size):
        # print("Selection")
        # sort agents by fitness
        agents = sorted(agents, key=lambda agent: agent.fitness, reverse=False)
        # select top agents
        agents = agents[:pop_size]
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
    def crossover(parent1, parent2, network):
        # print("Crossover")            
        # perform crossover
        child1 = Agent(network)
        child2 = Agent(network)
        shapes = [weight.shape for weight in parent1.network.weights]
        # perform crossover
        genes1 = np.concatenate([weight.flatten() for weight in parent1.network.weights])
        genes2 = np.concatenate([weight.flatten() for weight in parent2.network.weights])
        # randomly select the crossover point
        split = random.randint(0, genes1.size)
        child1_genes = np.concatenate([genes1[:split], genes2[split:]])
        child2_genes = np.concatenate([genes2[:split], genes1[split:]])
        # convert the flattened weights to the original shape
        child1.network.weights = GeneticAlgorithm.unflatten_weights(child1_genes, shapes)
        child2.network.weights = GeneticAlgorithm.unflatten_weights(child2_genes, shapes)
        return child1, child2

    # mutation function performs mutation on the agents by rate of 10%
    @staticmethod
    def mutation(agent, mute_rate):
        # print("Mutation")
        if random.uniform(0, 1) <= mute_rate:
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
        return agent

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
            # round to 0 or 1
            predictions = np.round(input_data)
        return predictions

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
def split_data(X, y, test_ratio=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def main():
    print("Hello World! with RWS")
    # # get input file path
    # input_file = input("Enter the file path: ")
    input_file = 'nn0.txt'
    # read the data from the file
    X, y = read_file(input_file)
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    # define the network
    network = [
        [16, 8, sigmoid],
        [None, 1, sigmoid]
    ]
    # define the parameters
    pop_size = 500
    generations = 1000
    # allow 0.005 error
    threshold = X_train.shape[0] * 0.005
    # execute the genetic algorithm
    best_network = GeneticAlgorithm.execute(pop_size, generations, threshold, X_train, y_train.T, network)
    # print(best_network.weights)
    # run the best network on the test data
    yhat = best_network.forward(X_test)
    # reshape y_hat to match y
    yhat = yhat.reshape(y_test.shape)
    # calculate fitness as the sum of absolute differences
    mistakes = np.sum(np.abs(yhat - y_test)) # calculate fitness as the sum of absolute differences
    print("Best Network Test Data Results:\nMistakes: ", mistakes,"\nAccuracy: ",(1 - round(mistakes / y_test.shape[0], 4)), "\n")
    


if __name__ == "__main__":
    main()
