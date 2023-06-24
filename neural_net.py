import numpy as np

from genetic_algo import genetic_train_nn
from utils import sigmoid, SampleClassification, train_test_split


class GeneticNeuralNet:
    """
    Implements the core algorithm of a simple NN model using Genetic Algorithms.

    NOTE: NN architecture must start with size 16 (with current samples) and end with size 1 (binary decision)
    """
    def __init__(self, neural_net_data, activation=sigmoid):
        """
        :param neural_net_data: neural net architecture & weights
        :type neural_net_data: neural_net_data.NeuralNetData
        :param activation: activation function to evaluate on net layers
        :type activation: function
        """
        self._net = neural_net_data
        self._activation = activation

    def train(self, samples, population_size=50, generations_count=1000):
        """
        Train the net using GA on a given set of samples
        :param samples: samples to train on
        :type samples: list[utils.SampleClassification]
        :param population_size: size of population in the GA
        :type population_size: int
        :param generations_count: number of generations to execute in the GA
        :type generations_count: int
        :return: success rate of the training process
        :rtype: float
        """
        test_samples, train_samples = train_test_split(samples, test_ratio=0.99)     # TODO: smaller ratio
        population = [self.random_copy() for _ in range(population_size)]
        best_model = genetic_train_nn(train_samples, population, generations_count)
        self._net = best_model._net

        inputs, classifications = zip(*test_samples)
        return self.success_rate(inputs, classifications)

    def fitness(self, inputs, classifications):
        """
        Calculate the fitness function of a given set of samples.
        :param inputs: samples to calculate loss on
        :type inputs: list[numpy.ndarray]
        :param classifications: samples classifications
        :type classifications: list[float]
        :return: total fitness value of given samples
        :rtype: float
        """
        evaluation = self._feed_forward(inputs)
        return np.sum(np.abs(evaluation - classifications)) / len(evaluation)

    def success_rate(self, inputs, classifications):
        """
        Calculate the loss function of a given set of samples.
        :param inputs: samples to calculate loss on
        :type inputs: list[numpy.ndarray]
        :param classifications: samples classifications
        :type classifications: list[float]
        :return: total loss value of the given samples
        :rtype: float
        """
        evaluation = self._feed_forward(inputs)
        return 1 - np.sum(np.abs(np.round(evaluation) - classifications)) / len(evaluation)

    def _feed_forward(self, inputs):
        results = []
        for input_data in inputs:
            layer_output = input_data
            for layer in self._net.layers:
                layer_output = layer_output @ layer.weights
                layer_output = self._activation(layer_output - 1)

            # The last layer must contain a single value
            results.append(layer_output[0])

        return np.array(results)

    def feed_forward(self, inputs):
        """
        Evaluate the model on a given data to get a final classification.
        :param inputs: data to evaluate model on.
        :type inputs: list[numpy.ndarray]
        :return: sample classification
        :rtype: list[utils.SampleClassification]
        """
        return [
            SampleClassification(input_data, round(classification))
            for input_data, classification in zip(inputs, self._feed_forward(inputs))
        ]

    def random_copy(self):
        """
        Copy the same NN architecture with uninitialized weights.
        :return: uninitialized NN
        :rtype: GeneticNeuralNet
        """
        return GeneticNeuralNet(
            activation=self._activation,
            neural_net_data=self._net.random_copy()
        )

    def copy(self):
        """
        Copy the same NN architecture.
        :return: uninitialized NN
        :rtype: GeneticNeuralNet
        """
        return GeneticNeuralNet(
            activation=self._activation,
            neural_net_data=self._net.copy()
        )

    @property
    def neural_net_data(self):
        return self._net
