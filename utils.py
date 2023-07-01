import random
import numpy as np
from collections import namedtuple


"""
ML Utils
"""


SampleClassification = namedtuple(
    "SampleClassification",
    [
        "data",             # type: np.ndarray
        "classification"    # type: int
    ]
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_test_split(samples, test_ratio=0.3):
    """
    Split a given set of samples into train samples and test samples by a defined ratio.
    :param samples: samples to split
    :type samples: list
    :param test_ratio: test ratio to choose for test samples
    :type test_ratio: float
    :return: test samples, train samples
    :rtype: tuple[list, list]
    """
    random.shuffle(samples)
    cut_ind = int(len(samples) * test_ratio)

    return samples[:cut_ind], samples[cut_ind:]


"""
IO Utils
"""


def _binary_string_to_np_array(binary_str):
    return np.array([int(d) for d in binary_str])


def read_samples_file(samples_file):
    """
    Read given classification samples file.
    :param samples_file: sample file to read from
    :type samples_file: str
    :return: samples classifications
    :rtype: list[SampleClassification]
    """
    with open(samples_file, "r") as input_file:
        raw_samples = input_file.readlines()

    samples = []
    for raw_sample in set(raw_samples):
        data, classification = raw_sample.strip().split()
        samples.append(
            SampleClassification(
                data=_binary_string_to_np_array(data.strip()),
                classification=int(classification)
            )
        )

    return samples


def read_data_file(data_file):
    """
    Read a given data file.
    :param data_file: data file to read from
    :type data_file: str
    :return: data
    :rtype: list[np.ndarray]
    """
    with open(data_file, "r") as input_file:
        return [
            _binary_string_to_np_array(data.strip())
            for data in input_file.readlines() if data
        ]


def write_results_file(results, results_file, show_data=False):
    """
    Write evaluated results into an output file.
    :param results: results to write
    :type results: list[utils.SampleClassification]
    :param results_file: results file path to write to
    :type results_file: str
    :param show_data: whether to show sample source data in results file lines.
    :type show_data: bool
    """
    with open(results_file, "w") as output_file:
        output_file.writelines([
            (f"{''.join(map(str, result.data))} " if show_data else "") +
            f"{result.classification}\n"
            for result in results
        ])
