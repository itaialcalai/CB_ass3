import argparse

from neural_net import GeneticNeuralNet
from neural_net_data import NeuralNetData, NeuralNetLayerData
from utils import read_samples_file


def build_net(samples_file, wnet_file):
    samples = read_samples_file(samples_file)
    neural_net_data = NeuralNetData([
        NeuralNetLayerData(input_size=16, output_size=1)
    ])

    neural_net = GeneticNeuralNet(neural_net_data)
    success_rate = neural_net.train(samples)
    print(f"Train success rate: {success_rate}")
    neural_net.neural_net_data.dump_to_file(wnet_file)


def main():
    parser = argparse.ArgumentParser(
        description="Generating NN model architecture for a given samples using GA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('samples_file', help='Input samples file to train & test on', default="nn1.txt")
    parser.add_argument('--wnet_file', help='Serialized generated NN model file to write to', default="wnet1.json")

    args = parser.parse_args()
    build_net(args.samples_file, args.wnet_file)


if __name__ == '__main__':
    main()
