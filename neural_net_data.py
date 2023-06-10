import json
import numpy as np


class NeuralNetLayerData:
    """
    Describes a simple layer data in the NN architecture.
    """
    def __init__(self, input_size, output_size, weights=None):
        """
        Initialize new layer data.
        If given weights parameter is None, random values will be initialized.
        :param input_size: layer input size
        :type input_size: int
        :param output_size: layer output size
        :type output_size: int
        :param weights: layer weights
        :type weights: numpy.ndarray
        """
        self._input_size = input_size
        self._output_size = output_size
        self._weights = np.random.randn(input_size, output_size) if weights is None else weights

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

    @property
    def weights(self):
        return self._weights


class NeuralNetData:
    """
    Describes the actual structure & weights of a simple NN with a single activation function as sigmoid.
    """
    def __init__(self, layers):
        """
        Initialize new NN data.
        If weights is not initialized (None), random weight will be generated once.
        :type layers: list[NeuralNetLayerData]
        """
        self._layers = layers

    def random_copy(self):
        return NeuralNetData([
            NeuralNetLayerData(layer.input_size, layer.output_size)
            for layer in self._layers
        ])

    def dump_to_file(self, file_path):
        layers = [
            (layer.input_size, layer.output_size, layer.weights.tolist())
            for layer in self._layers
        ]

        with open(file_path, "w") as output_file:
            return json.dump(layers, output_file)

    @classmethod
    def load_from_file(cls, file_path):
        with open(file_path, "r") as input_file:
            layers_data = json.load(input_file)

        return cls([
            NeuralNetLayerData(input_size, output_size, np.asarray(weights))
            for input_size, output_size, weights in layers_data
        ])

    @property
    def layers(self):
        return self._layers

    def __len__(self):
        return len(self._layers)
