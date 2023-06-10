from collections import namedtuple


_Agent = namedtuple(
    "Agent",
    [
        "fitness",
        "neural_net"
    ]
)


def genetic_train_nn(samples, population, generations_count, mutation_rate=0.8, fitness_threshold=0.01, stuck_limit=50):
    """
    Search for the best NN model weights using a GA.
    :param samples: samples to learn on
    :type samples: list[utils.SampleClassification]
    :param population: NN model genetic population
    :type population: list[neural_net.GeneticNeuralNet]
    :param generations_count: maximal number of generation to execute
    :type generations_count: int
    :param mutation_rate: rate of mutant creation during the genetic algorithm
    :type mutation_rate: float
    :param fitness_threshold: sufficient fitness value to stop generation evaluation
    :type fitness_threshold: float
    :param stuck_limit: number of times to break after unchanged fitness
    :type stuck_limit: int
    :return: best NN model found so far
    :rtype: neural_net.GeneticNeuralNet
    """
    inputs, classifications = zip(*samples)
    population = [_Agent(0, nn) for nn in population]
    interval = generations_count / 4            # TODO
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


        # TODO: selection & mutation & crossover
        """
        # decrease the mutation rate over time
        if generation % interval == 0:
            mutation_rate /= 2

        agents = GeneticAlgorithm.selection(agents)

        # crossover and mutation
        agents = GeneticAlgorithm.crossover(agents, pop_size, network)
        agents = GeneticAlgorithm.mutation(agents, mutation_rate)
        """

    return best_agent.neural_net
