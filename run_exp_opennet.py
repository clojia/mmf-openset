import argparse
import subprocess
import random
import os
import tensorflow as tf
import sys
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
from tensorflow.python.client import device_lib



def main():
    parser = argparse.ArgumentParser(description='OpenNetFlat experiments.')
    parser.add_argument('-eid', '--exp_id', required=False, dest='exp_id',
                        default=None, help='path to output directory.')
    parser.add_argument('-n','--network', required=True, dest='network',
                        choices=['flat', 'cnn'], help='dataset name.')
    parser.add_argument('-ds','--datasets', required=True, dest='dataset_name',
                        choices=['mnist', 'ms', 'android', 'msadjmat'], help='dataset name.')
    parser.add_argument('-m','--models', required=True, dest='model_names', nargs='+',
                        default=['ii', 'ce', 'ceii', 'openmax', 'g_openmax', 'central', 'cor', 'triplet', 'tripletcor', 'ceiicor', 'cecor', 'iicor'], help='model name.')
    parser.add_argument('-trc_file', '--tr_classes_list_file', required=True, dest='trc_file',
                        help='list of training classes.')
    parser.add_argument('-o', '--outdir', required=False, dest='output_dir',
                        default='./exp_result/cnn', help='path to output directory.')
    parser.add_argument('-s', '--seed', required=False, dest='seed', type=int,
                        default=1, help='path to output directory.')
    parser.add_argument('--closed', dest='closed', action='store_true',
                        help='Run closed world experiments.')
    parser.add_argument('--no-closed', dest='closed', action='store_false',
                        help='Run open world experiments.')
    parser.set_defaults(closed=False)

    args = parser.parse_args()

    if args.exp_id is None:
        args.exp_id = random.randint(0, 10000)

    tr_classes_list = []
    with open(args.trc_file) as fin:
        for line in fin:
            if line.strip() == '':
                continue
            cols = line.strip().split()
            tr_classes_list.append([int(float(c)) for c in cols])

    for tr_classes in tr_classes_list:
        for mname in args.model_names:
            exp_args = []
            exp_args += ['python', 'exp_opennet.py']
            exp_args += ['-e', str(args.exp_id)]
            exp_args += ['-n', args.network]
            exp_args += ['-m', mname]
            exp_args += ['-ds', args.dataset_name]
            exp_args += ['-trc']
            exp_args += [str(c) for c in tr_classes[:10]]
            exp_args += ['-o', args.output_dir]
            exp_args += ['-s', str(args.seed)]
            if args.closed:
                exp_args += ['--closed']

            print(exp_args)
            proc = subprocess.Popen(exp_args)
            proc.wait()

if __name__ == '__main__':
    main()
