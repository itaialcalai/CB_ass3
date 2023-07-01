# CB_ass3 #
Recognize patterns in binary strings using neural networks and a GA for training the model.


## Installation ##
No special installation is required :)

Please note that the `numpy` package must be installed on your python env. 
```commandline
python -m pip install numpy
```

## Usage ##
Splitting samples data into train set and test set.
```commandline
python split_data.py -h  // help message
python split_data.py nn0.txt nn0_train.txt nn0_test.txt  // split to output files
```

Building the NN using GA.
```commandline
python buildnet0.py -h  // help message
python buildnet0.py nn0_train.txt nn0_test.txt  // -> wnet0.txt
```

Calculating success rate of a given model.
```commandline
python success_rate.py -h  // help message
python success_rate.py nn0_test.txt wnet0.txt  // print success rate
```

Evaluating the NN.
```commandline
python runnet0.py -h  // help message
python runnet0.py wnet0.txt samples0.txt // -> results.txt
```

## Remarks ##
* Currently, NN architecture is constant per execution and not calculated as part of the GA (but may be easily configured).
