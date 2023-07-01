import argparse

from utils import read_samples_file
from neural_net import GeneticNeuralNet
from neural_net_data import NeuralNetData


def eval_success_rate(data_file, wnet_file):
    samples = read_samples_file(data_file)
    neural_net_data = NeuralNetData.load_from_file(wnet_file)
    neural_net = GeneticNeuralNet(neural_net_data)

    inputs, classifications = zip(*samples)
    success_rate = neural_net.success_rate(inputs, classifications)
    print(f"Success Rate: {success_rate}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluating a given model on classified samples file for success rate calculation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data_file', help='Input data file to evaluate model on', default="nn.txt")
    parser.add_argument('wnet_file', help='Serialized NN model file to evaluate on input data', default="wnet.txt")

    args = parser.parse_args()
    eval_success_rate(args.data_file, args.wnet_file)


if __name__ == '__main__':
    main()
