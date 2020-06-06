# MMF: A loss extension for feature learning in openset recognition

This repository is the official implementation of "MMF: A loss extension for feature learning in openset recognition". 

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```


## Training and Testing

To run the experiments in the paper, run this command:

```train
python run_exp_opennet.py -n <network_type>[cnn, flat] -ds <dataset>[mnist, msadjmat, android] -m <loss_function_type>[ce, cemmf, ii, iimmf, triplet, tripletmmf] -trc_file <known_classes_file>[mnist_trc_list, msadjmat_trc_list, android_trc_list] -o <output_file>
```
e.g. 
```
# For MNIST dataset
python run_exp_opennet.py -n cnn -ds mnist -m ce cemmf ii iimmf triplet tripletmmf -trc_file "data/mnist_trc_list" -o "data/results/cnn/mnist"

# For AG dataset
python run_exp_opennet.py -n flat -ds android -m ce cemmf ii iimmf triplet tripletmmf -trc_file "data/android_trc_list" -o "data/results/cnn/android"

# For MC dataset
python run_exp_opennet.py -n cnn -ds msadjmat -m ce cemmf ii iimmf triplet tripletmmf -trc_file "data/msadjmat_trc_list" -o "data/results/cnn/msadjmat"
```


## Evaluation

To evaluate my model on ImageNet, use the notebook "mmf_metrics", replace the directory with the output directory.

## Results


## Contributing
This project is licensed under the terms of the MIT license.
