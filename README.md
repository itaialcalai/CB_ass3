# CB_ass3 #
Recognize patterns in binary strings using neural networks and a GA for training the model.


## Installation ##
No special installation is required :)

Please note that the `numpy` package must be installed on your python env. 
```commandline
python -m pip install numpy
```

## Usage ##
Building the NN using GA.
```commandline
python buildnet0.py -h
python buildnet0.py nn0.txt  // -> wnet.json
```

Evaluating the NN.
```commandline
python runnet0.py -h
python runnet0.py samples0.txt // -> results.txt
```

## Remarks ##
* Currently, NN architecture is constant per execution and not calculated as part of the GA.