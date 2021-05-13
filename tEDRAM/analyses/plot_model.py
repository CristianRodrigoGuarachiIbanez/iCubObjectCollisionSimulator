"""
    keras plot_model wrapper

"""
import os
from argparse import ArgumentParser
from keras.utils import plot_model

from models.model_keras import *
from config import config


def main(gpu_id, model_id, plot_name, plot_recurrent, plot_output_shapes,
         n_steps, glimpse_size, coarse_size, conv_sizes, n_filters, fc_dim,
         enc_dim, dec_dim, n_classes, output_mode, use_init_matrix,
         output_emotion_dims, use_batch_norm, dropout):


    save_path = '/scratch/forch/output_git/' + plot_name

    glimpse_size = (glimpse_size, glimpse_size)
    coarse_size = (coarse_size, coarse_size)

    print("\n[Info] Using GPU", gpu_id)
    if gpu_id == -1:
        print('[Error] You need to select a gpu. (e.g. python train.py --gpu=7)\n')
        exit()
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if model_id==1:
        model = edram_model(config['input_shape'], 1, n_steps,
                            glimpse_size, coarse_size, hidden_init=0,
                            n_filters=n_filters, filter_sizes=(3,5), n_features=fc_dim,
                            RNN_size_1=enc_dim, RNN_size_2=dec_dim, n_classes=n_classes,
                            output_mode=output_mode, use_init_matrix=use_init_matrix,
                            output_emotion_dims=output_emotion_dims,
                            bn=use_batch_norm, dropout=dropout)
    elif model_id==2:
        model = edram_model_2(config['input_shape'], 1, n_steps,
                            glimpse_size, coarse_size, hidden_init=0,
                            n_filters=n_filters, filter_sizes=(3,5), n_features=fc_dim,
                            RNN_size_1=enc_dim, RNN_size_2=dec_dim, n_classes=n_classes,
                            output_mode=output_mode, use_init_matrix=use_init_matrix,
                            output_emotion_dims=output_emotion_dims,
                            bn=use_batch_norm, dropout=dropout)
    elif model_id==3:
        output_localisation = False if output_mode==0 and n_steps==1 else True
        model = edram_cell(config['input_shape'], glimpse_size, n_filters,
                       (3,3), fc_dim, enc_dim, dec_dim,
                       n_classes, use_batch_norm, dropout, 1, Dense(6, name='emission'),
                       output_localisation, output_emotion_dims)
    else:
        print('[Error] Only model 1 to 3 are available!\n')
        exit()

    # , expand_nested=plot_recurrent is missing
    model.summary()
    plot_model(model, to_file=save_path, show_shapes=plot_output_shapes)

    print("\n Saved model plot to: " + save_path)

    exit()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--gpu", type=int, nargs='?', default=-1, const=7,
                        dest='gpu_id', help="Specifies the GPU.")
    # plot parameters
    parser.add_argument("--name", type=str, default='model.png',
                            dest='plot_name', help="Save path for the plot.")
    parser.add_argument("--recurrent", type=int, nargs='?', default=0, const=1,
                            dest='plot_recurrent', help="Save path for the plot.")
    parser.add_argument("--shapes", type=int, nargs='?', default=0, const=1,
                            dest='plot_output_shapes', help="Save path for the plot.")
    # model parameters
    parser.add_argument("--model", type=int, default=1,
                        dest='model_id', help="Selects model type.")
    parser.add_argument("--steps", type=int, default=3,
                        dest="n_steps", help="Step size for digit recognition.")
    parser.add_argument("--glimpse_size", "-a", type=int, default=26,
                        dest='glimpse_size', help="Window size of attention mechanism.")
    parser.add_argument("--coarse_size", type=int, default=12,
                        dest='coarse_size', help="Size of the rescaled input image for initialization of the network.")
    parser.add_argument("--conv_sizes", type=int, nargs='+', default=[3, 5],
                        dest="conv_sizes", help="List of sizes of convolution filters.")
    parser.add_argument("--conv_filters", type=int, default=128,
                        dest="n_filters", help="Number of filters in convolution.")
    parser.add_argument("--fc_dim", type=int, default=1024,
                        dest="fc_dim", help="Fully connected dimension.")
    parser.add_argument("--enc_dim", type=int, default=512,
                        dest="enc_dim", help="Encoder RNN state dimension.")
    parser.add_argument("--dec_dim", type=int, default=512,
                        dest="dec_dim", help="Decoder  RNN state dimension.")
    parser.add_argument("--classes", type=int, default=10,
                        dest="n_classes", help="Number of classes for recognition.")
    parser.add_argument("--mode", "--output_mode", type=int, default=1,
                          dest="output_mode", help="Output last step or all steps.")
    parser.add_argument("--use_init", "--use_init_matrix", type=int, default=1,
                          dest="use_init_matrix", help="Whether to use the init matrix as output.")
    parser.add_argument("--output_dims", "--output_emotion_dims", type=int, default=0,
                          dest="output_emotion_dims", help="Whether to output valence and arousal.")
    parser.add_argument("--bn", "--use_batch_norm", type=int, default=1,
                        dest="use_batch_norm", help="Whether to use batch normalization.")
    parser.add_argument("--do", "--dropout", type=float, default=0,
                        dest="dropout", help="Whether to use dropout (dropout precentage).")
    args = parser.parse_args()


    main(**vars(args))
