import random
import numpy as np
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
        fitness_threshold=0.02,
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
    last_fitness, stuck_state = None, 0
    best_agent = None

    for generation in range(generations_count):
        # TODO: performance issues
        population_agents = sorted(
            [_Agent(neural_net.fitness(inputs, classifications), neural_net)
            for neural_net in population],
            key=lambda a: a.fitness
        )
        best_agent = population_agents[0]
        stuck_state = (0 if last_fitness != best_agent.fitness else stuck_state+1)
        last_fitness = best_agent.fitness
        print(f"#{generation} best fitness {best_agent.fitness}")

        if best_agent.fitness <= fitness_threshold:
            print(f"#{generation} Fitness threshold met")
            return best_agent.neural_net

        if stuck_state == stuck_limit:
            print(f"#{generation} stuck limit met")
            return best_agent.neural_net

        # TODO: change rates over time? decrease mutations?
        population = _generate_next_generation(population_agents, selection_rate, mutation_rate, crossover_rate)

    return best_agent.neural_net


def _generate_next_generation(population_agents, selection_rate, mutation_rate, crossover_rate):
    selection_count = int(selection_rate * len(population_agents))
    mutation_count = int(mutation_rate * len(population_agents))
    crossover_count = int(crossover_rate * len(population_agents))
    randomize_count = len(population_agents) - selection_count - mutation_count - crossover_count

    selection_population = [agent.neural_net for agent in population_agents[:selection_count]]
    mutation_population = _generate_mutations(population_agents, mutation_count)
    crossover_population = _generate_crossovers(population_agents, crossover_count)
    randomized_population = [
        population_agents[0].neural_net.random_copy()
        for _ in range(randomize_count)
    ]

    return selection_population + mutation_population + crossover_population + randomized_population


def _mutate(neural_net, mutation_rate):
    mute_net = neural_net.copy()
    for layer in mute_net.neural_net_data.layers:
        if random.random() < mutation_rate:
            layer.weights += np.random.normal(0, 0.07, layer.weights.shape)
    return mute_net


def _generate_mutations(population_agents, mutation_count):
    mutated_pop = []
    for agent in population_agents[:mutation_count]:
        mutated_net = _mutate(agent.neural_net, 0.1)
        mutated_pop.append(mutated_net)
    return mutated_pop


def _crossover(first, second):
    cross_net = first.copy()
    for i in range(len(cross_net.neural_net_data.layers)):
        cross_net.neural_net_data.layers[i].weights += second.neural_net_data.layers[i].weights
        cross_net.neural_net_data.layers[i].weights /= 2

    return cross_net


def _generate_crossovers(population_agents, crossover_count):
    crossover_pop = []
    for i in range(crossover_count):
        first = random.choice(population_agents)
        second = random.choice(population_agents)

        cross_net = _crossover(first.neural_net, second.neural_net)
        crossover_pop.append(cross_net)

    return crossover_pop
