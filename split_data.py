import argparse

from utils import read_samples_file, write_results_file, train_test_split


def split_data_file(data_file, train_file, test_file):
    samples = read_samples_file(data_file)
    test_samples, train_samples = train_test_split(samples)
    write_results_file(test_samples, test_file, show_data=True)
    write_results_file(train_samples, train_file, show_data=True)


def main():
    parser = argparse.ArgumentParser(
        description="Split data into training samples set and testing samples set given an input data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data_file', help='Input data file with all samples', default="nn.txt")
    parser.add_argument('--train_file', help='Output training samples set file', default="nn_train.txt")
    parser.add_argument('--test_file', help='Output testing samples set file', default="nn_test.txt")

    args = parser.parse_args()
    split_data_file(args.data_file, args.train_file, args.test_file)


if __name__ == '__main__':
    main()
