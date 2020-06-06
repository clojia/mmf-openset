# MMF: A loss extension for feature learning in openset recognition

This repository is the official implementation of "MMF: A loss extension for feature learning in openset recognition". 

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```


## Training and Evaluatior
The experiments will load the dataset from the `data/` directory, then split it into training, validation and test sets in customized ratio. The default ratio (also the ratio we use in the paper) is `0.75(training): 0.25(testing)`. To run the experiments in the paper, use this command:

```train
python run_exp_opennet.py -n <network_type>[cnn, flat] -ds <dataset>[mnist, msadjmat, android] -m <loss_function_type>[ce, cemmf, ii, iimmf, triplet, tripletmmf] -trc_file <known_classes_file>[mnist_trc_list, msadjmat_trc_list, android_trc_list] -o <output_file>
```
e.g. 
```
# For MNIST dataset
python run_exp_opennet.py -n cnn -ds mnist -m ce cemmf ii iimmf triplet tripletmmf -trc_file "data/mnist_trc_list" -o "data/results/cnn/mnist"

# For AG dataset
python run_exp_opennet.py -n flat -ds android -m ce cemmf ii iimmf triplet tripletmmf -trc_file "data/android_trc_list" -o "data/results/flat/android"

# For MC dataset
python run_exp_opennet.py -n cnn -ds msadjmat -m ce cemmf ii iimmf triplet tripletmmf -trc_file "data/msadjmat_trc_list" -o "data/results/cnn/msadjmat"
```
The model will be saved as `.cptk` files under customized model directory.

## Results
AUC:
|         | CE  | CE+MMF | II | II+MMF | Triplet | Triplet+MMF |
| --------|-----| ----- |----- |----------- | -----|--------- |
| MNIST   |  0.9255  |  0.9479   | 0.9578 | 0.9649 | 0.9496 | 0.9585 |
| Microsoft Challenge | 0.9148 | 0.9500 | 0.9385 | 0.9461 | 0.9240 | 0.9430 | 
| Android Genome | 0.7506 | 0.8205 | 0.8427 | 0.8694 | 0.8271 | 0.8379 |

F1:
|         | CE  | CE+MMF | II | II+MMF | Triplet | Triplet+MMF |
| --------|-----| ----- |----- |----------- | -----|--------- |
| MNIST   |  0.7591  |  0.8809   | 0.9250 | 0.9308 | 0.8989 | 0.8989 |
| Microsoft Challenge | 0.8562 | 0.8929 | 0.8860 | 0.8991 | 0.8715 | 0.8800 | 
| Android Genome | 0.4587 | 0.4925 | 0.6396 | 0.6528 | 0.5986 | 0.6132 |

All the evaluation results will be saved as `.pkl` files once experiments are done. To load the results, use the notebook `mmf_metrics.ipynb`, replace the directory with the output directory.

## Contributing
If you'd like to contribute, or have any suggestions for these guidelines, you can open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license.
