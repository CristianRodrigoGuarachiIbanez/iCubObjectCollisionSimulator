"""
    Fixes a compatibility issue when training a model with use_init=0 and output_mode=0

    Copies the emission network weights from the edram cell to the edram model

    or

    shows the elements of the model_weights.h5

"""

from __future__ import print_function
import os
import h5py
import numpy as np

from argparse import ArgumentParser

def main(path, mode):

    if mode==0:

        for weights in ['model_weights.h5', 'checkpoint_weights.h5']:

            # get weights
            f = h5py.File(path + weights, 'r+')
            e = f['edram_cell']
            em = e['emission']
            bias = em['bias:0']
            weights = em['kernel:0']

            # copy weights
            em_new = f.create_group('emission/emission')
            bias_new = em_new.create_dataset('bias:0', data=bias[:])
            weights_new = em_new.create_dataset('kernel:0', data=weights[:,:])

            f.close()
    else:

        f = h5py.File(path + 'model_weights.h5', 'r')

        for i in f:
            print(i)
            for j in f[str(i)]:
                print('  ', j)
                for k in f[str(i)+'/'+str(j)]:
                    print('     ',k)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default='.',
                            dest='path', help="Path to the model weights.")
    parser.add_argument("--mode", type=int, default=0,
                            dest='mode', help="Copy (0) or Print (1)")
    args = parser.parse_args()

    main(**vars(args))
