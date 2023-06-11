from collections import namedtuple


_Agent = namedtuple(
    "Agent",
    [
        "fitness",
        "neural_net"
    ]
)


def genetic_train_nn(
        samples,
        population,
        generations_count,
        selection_rate=0.1,
        mutation_rate=0.4,
        crossover_rate=0.4,
        fitness_threshold=0.01,
        stuck_limit=50
):
    """
    Search for the best NN model weights using a GA.
    :param samples: samples to learn on
    :type samples: list[utils.SampleClassification]
    :param population: NN model genetic population
    :type population: list[neural_net.GeneticNeuralNet]
    :param generations_count: maximal number of generation to execute
    :type generations_count: int
    :param selection_rate: rate of keeping top population during the genetic algorithm
    :type selection_rate: float
    :param mutation_rate: rate of mutant creation during the genetic algorithm
    :type mutation_rate: float
    :param crossover_rate: rate of crossover creation during the genetic algorithm
    :type crossover_rate: float
    :param fitness_threshold: sufficient fitness value to stop generation evaluation
    :type fitness_threshold: float
    :param stuck_limit: number of times to break after unchanged fitness
    :type stuck_limit: int
    :return: best NN model found so far
    :rtype: neural_net.GeneticNeuralNet
    """
    inputs, classifications = zip(*samples)
    population = [_Agent(0, nn) for nn in population]
    last_fitness, stuck_state = None, 0
    best_agent = population[0]

    for generation in range(generations_count):
        # TODO: performance issues
        population = sorted(
            _Agent(agent.neural_net.fitness(inputs, classifications), agent.neural_net)
            for agent in population
        )
        best_agent = population[0]
        stuck_state = (0 if last_fitness != best_agent.fitness else stuck_state+1)
        print(f"#{generation} best fitness {best_agent.fitness}")

        if best_agent.fitness <= fitness_threshold:
            print(f"#{generation} Fitness threshold met")
            return best_agent.neural_net

        if stuck_state == stuck_limit:
            print(f"#{generation} stuck limit met")
            return best_agent.neural_net

        # TODO: change rates over time? decrease mutations?
        population = _generate_next_generation(population, selection_rate, mutation_rate, crossover_rate)

    return best_agent.neural_net


def _generate_next_generation(population, selection_rate, mutation_rate, crossover_rate):
    selection_count = int(selection_rate * len(population))
    mutation_count = int(mutation_rate * len(population))
    crossover_count = int(crossover_rate * len(population))
    randomize_count = len(population) - selection_count - mutation_count - crossover_count

    selection_population = population[:selection_count]
    mutation_population = _generate_mutations(population, mutation_count)
    crossover_population = _generate_crossovers(population, crossover_count)
    randomized_population = [
        _Agent(0, population[0].neural_net.random_copy())
        for _ in range(randomize_count)
    ]

    return selection_population + mutation_population + crossover_population + randomized_population


def _generate_mutations(population, mutation_count):
    # TODO: impl
    return population[:mutation_count]


def _generate_crossovers(population, crossover_count):
    # TODO: impl
    return population[:crossover_count]

"""
    # roulette wheel selection method
    @staticmethod
    def roulette_wheel_selection(agents):
        fitness_values = [agent.fitness for agent in agents]  # Get fitness values from agents
        cumulative_fitness = np.cumsum(fitness_values)  # Calculate the cumulative sum of fitness values
        selection_point = sum(fitness_values) * np.random.rand()  # Generate a random selection point
        index = np.argwhere(selection_point <= cumulative_fitness)[0][0]  # Find the index where selection_point falls
        return index  # Return the selected individual

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
        # print("Crossover")
        offspring = []
        for _ in range((pop_size - len(agents)) // 2):
            # randomly select two parents -> TODO: roulette wheel selection
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
    def mutation(agents, mute_rate):
        # print("Mutation")
        for agent in agents:
            if random.uniform(0, 1) <= 0.2:
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

"""