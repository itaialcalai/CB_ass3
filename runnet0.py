import argparse

from neural_net import GeneticNeuralNet
from neural_net_data import NeuralNetData
from utils import read_data_file, write_results_file


def run_net(wnet_file, data_file, results_file):
    neural_net_data = NeuralNetData.load_from_file(wnet_file)
    data = read_data_file(data_file)

    neural_net = GeneticNeuralNet(neural_net_data)
    results = neural_net.feed_forward(data)
    write_results_file(results, results_file)


def main():
    parser = argparse.ArgumentParser(
        description="Executing serialized NN architecture on a given input data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('wnet_file', help='Serialized NN model file to evaluate on input data', default="wnet.json")
    parser.add_argument('data_file', help='Input data file to evaluate model on', default="nn.txt")
    parser.add_argument('--results_file', help='Output file to write final classifications', default="result.txt")

    args = parser.parse_args()
    run_net(args.wnet_file, args.data_file, args.results_file)


if __name__ == '__main__':
    main()
