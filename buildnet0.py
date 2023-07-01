import argparse

from neural_net import GeneticNeuralNet
from neural_net_data import NeuralNetData, NeuralNetLayerData
from utils import read_samples_file


def build_net(train_file, test_file, wnet_file):
    train_samples = read_samples_file(train_file)
    test_samples = read_samples_file(test_file)
    neural_net_data = NeuralNetData([
        NeuralNetLayerData(input_size=17, output_size=1)
    ])

    neural_net = GeneticNeuralNet(neural_net_data)
    success_rate = neural_net.train(train_samples, test_samples)
    print(f"Test success rate: {success_rate}")
    neural_net.neural_net_data.dump_to_file(wnet_file)


def main():
    parser = argparse.ArgumentParser(
        description="Generating NN model architecture for a given samples using GA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('train_file', help='Input samples file to train on', default="nn_train.txt")
    parser.add_argument('test_file', help='Input samples file to test on', default="nn_test.txt")
    parser.add_argument('--wnet_file', help='Serialized generated NN model file to write to', default="wnet0.txt")

    args = parser.parse_args()
    build_net(args.train_file, args.test_file, args.wnet_file)


if __name__ == '__main__':
    main()
